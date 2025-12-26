import os, sys, json, time, math, gc, re, io, signal, random, subprocess
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

def _pip_install(pkgs: List[str]):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

try:
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login, whoami
    
except Exception:
    _pip_install([
        "transformers==4.46.2",
        "accelerate==0.34.2",
        "sentencepiece",
        "datasets==2.20.0",
        "pandas==2.2.2",
        "pyarrow<20",
        "scikit-learn",
        "matplotlib",
        "huggingface_hub",
        "tqdm",
    ])
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login, whoami

from tqdm import tqdm

print("torch:", torch.__version__)

@dataclass
class Config:
    out_dir: str = "uncertainty_runs"
    max_tasks: int = 50             
    n_prompts: int = 5              
    n_val_tasks: int = 10           
    max_new_tokens: int = 256

    temp_attempt0: float = 0.0
    top_p_attempt0: float = 1.0
    temp_regen: float = 0.6
    top_p_regen: float = 0.92

    pfail_threshold: float = 0.50
    max_attempts: int = 3  

    seed: int = 42
    use_cache: bool = True

CFG = Config()

MODELS = [
    ("llama", "3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("qwen-coder", "3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
]

BAYESPE_TEMPLATES = [
    "You are an expert Python developer. Write only the function implementation.\n\n{p}\n# Your solution:\n",
    "Complete the Python function so it passes the tests. No explanations.\n\n{p}\n# Implementation:\n",
    "Return a minimal and correct Python solution for this function.\n\n{p}\n# Code:\n",
    "Implement the function described below in Python. Output only valid code.\n\n{p}\n# Solution:\n",
    "Write a concise Python implementation that satisfies the specification.\n\n{p}\n# Function:\n",
]

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass("Paste your Hugging Face token: ").strip()

login(HF_TOKEN, add_to_git_credential=False)
print("Logged in as:", whoami().get("name", "unknown"))
os.makedirs(CFG.out_dir, exist_ok=True)

def model_key(family: str, size: str, model_id: str) -> str:
    safe = model_id.replace("/", "__")
    return f"{family}__{size}__{safe}"

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def jload(path):
    with open(path, "r") as f:
        return json.load(f)

def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def load_jsonl_as_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```python\s*\n", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"^```\s*\n", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()

def _run_test_with_timeout(module_src: str, entry_point: str, timeout_seconds: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Executes HumanEval prompt+candidate+tests with timeout.
    Warning: exec is unsafe in general. Run in an isolated environment.
    """
    f = io.StringIO()
    use_timeout = hasattr(signal, "SIGALRM") and os.name != "nt"
    old_handler = None

    if use_timeout:
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test execution exceeded {timeout_seconds} seconds")
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        except Exception:
            use_timeout = False

    try:
        with redirect_stdout(f), redirect_stderr(f):
            glb = {}
            exec(module_src, glb, glb)
            cand = glb[entry_point]
            glb["candidate"] = cand
            glb["check"](cand)
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if use_timeout and old_handler is not None:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

def humaneval_pass(prompt: str, code: str, test_src: str, entry_point: str) -> Tuple[bool, Optional[str]]:
    code = strip_markdown_fences(code)
    if not code:
        return False, "empty_code"
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    module_src = prompt + "\n" + code + "\n\n" + test_src
    return _run_test_with_timeout(module_src, entry_point, timeout_seconds=10)

def load_model(model_id: str, hf_token: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model

def build_prompt_for_model(raw_prompt: str, family: str) -> Dict[str, Any]:
    system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
    user = raw_prompt + "\n\n# Your code below:\n"
    return {"chat": True, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

def token_entropies_from_logits_list(scores: List[torch.Tensor]) -> Tuple[List[float], float]:
    H = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        H.append(ent)
    return H, (float(np.mean(H)) if H else float("nan"))

@torch.inference_mode()
def generate_with_entropy(tok, model, prompt_text: str, family: str,
                          max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    device = next(model.parameters()).device
    spec = build_prompt_for_model(prompt_text, family)
    has_chat_template = getattr(tok, "chat_template", None) not in (None, "")

    if spec["chat"] and has_chat_template:
        input_ids = tok.apply_chat_template(spec["messages"], add_generation_prompt=True, return_tensors="pt").to(device)
        attention_mask = None
    else:
        parts = []
        for msg in spec["messages"]:
            parts.append(f"[{msg['role'].upper()}] {msg['content']}\n")
        text = "".join(parts) + "\n[ASSISTANT]\n"
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature != 0.0),
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.eos_token_id,
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    out = model.generate(**gen_kwargs)
    gen_ids = out.sequences[0, input_ids.shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    H_list, H_mean = token_entropies_from_logits_list(out.scores)
    return {
        "generated_code": strip_markdown_fences(gen_text),
        "mean_token_entropy_nats": H_mean,
        "first5_entropies": H_list[:5],
        "num_steps": len(H_list),
        "num_chars": len(gen_text),
    }

def apply_mte_prompt(raw_prompt: str, idx: int) -> str:
    tpl = BAYESPE_TEMPLATES[idx % len(BAYESPE_TEMPLATES)]
    return tpl.format(p=raw_prompt.strip())

@dataclass
class PfailModel:
    w1: float
    w2: float
    b: float
    H_mean: float
    H_std: float
    V_mean: float
    V_std: float

def fit_pfail_model(H: np.ndarray, V: np.ndarray, y_fail: np.ndarray) -> PfailModel:
    H_mean = float(H.mean())
    H_std  = float(H.std() if H.std() > 1e-8 else 1.0)
    V_mean = float(V.mean())
    V_std  = float(V.std() if V.std() > 1e-8 else 1.0)

    X = np.stack([(H - H_mean)/H_std, (V - V_mean)/V_std], axis=1)

    if len(np.unique(y_fail)) > 1:
        lr = LogisticRegression()
        lr.fit(X, y_fail)
        w1, w2 = lr.coef_[0]
        b = lr.intercept_[0]
    else:
        p = float(y_fail.mean())
        p = min(max(p, 1e-4), 1 - 1e-4)
        w1, w2 = 0.0, 0.0
        b = float(np.log(p/(1-p)))

    return PfailModel(
        w1=float(w1), w2=float(w2), b=float(b),
        H_mean=H_mean, H_std=H_std, V_mean=V_mean, V_std=V_std
    )

def pfail(pmodel: PfailModel, H: float, V: float) -> float:
    Hs = (H - pmodel.H_mean) / pmodel.H_std
    Vs = (V - pmodel.V_mean) / pmodel.V_std
    logit = pmodel.w1*Hs + pmodel.w2*Vs + pmodel.b
    return float(1.0 / (1.0 + math.exp(-logit)))

def plot_pfail_hist_and_scatter(df: pd.DataFrame, out_prefix: str):
    d = df.dropna(subset=["pfail", "passed"]).copy()
    if len(d) == 0:
        print("No data to plot for", out_prefix)
        return

    passed = d["passed"].astype(int).values
    pf = d["pfail"].astype(float).values

    pf_pass = pf[passed == 1]
    pf_fail = pf[passed == 0]

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 25)
    if len(pf_pass):
        plt.hist(pf_pass, bins=bins, alpha=0.6, label=f"Correct (n={len(pf_pass)})", edgecolor="black", linewidth=0.5)
    if len(pf_fail):
        plt.hist(pf_fail, bins=bins, alpha=0.6, label=f"Incorrect (n={len(pf_fail)})", edgecolor="black", linewidth=0.5)
    plt.xlabel("Uncertainty Pfail")
    plt.ylabel("Frequency")
    plt.title("Pfail distribution (blue=correct, red=incorrect)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    png1 = out_prefix + "_pfail_hist.png"
    plt.savefig(png1, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    jitter = np.random.normal(0, 0.01, size=len(pf))
    x = np.clip(pf + jitter, 0, 1)
    colors = ["blue" if p == 1 else "red" for p in passed]
    plt.scatter(x, passed, alpha=0.6, s=40, edgecolors="black", linewidths=0.4, c=colors)
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.02, 1.02)
    plt.xlabel("Uncertainty Pfail")
    plt.ylabel("Correctness")
    plt.title("Pfail vs correctness (jittered x)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    png2 = out_prefix + "_pfail_scatter.png"
    plt.savefig(png2, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", png1)
    print("Saved:", png2)

def evaluate_one_model(family: str, size: str, model_id: str, dataset: List[Dict[str, Any]]):
    mk = model_key(family, size, model_id)
    mdir = os.path.join(CFG.out_dir, mk)
    os.makedirs(mdir, exist_ok=True)

    weights_path = os.path.join(mdir, "pfail_weights.json")
    tasklog_path  = os.path.join(mdir, "task_results.jsonl")
    df_csv_path   = os.path.join(mdir, "results.csv")

    done = {}
    if CFG.use_cache and os.path.exists(tasklog_path):
        for r in load_jsonl_as_list(tasklog_path):
            done[(r["task_id"])] = r

    print("\nLoading model:", model_id)
    t0 = time.time()
    tok, model = load_model(model_id, hf_token=HF_TOKEN)
    print(f"Loaded in {time.time() - t0:.1f}s")

    val_items = dataset[:min(CFG.n_val_tasks, len(dataset))]

    if CFG.use_cache and os.path.exists(weights_path):
        pmodel = PfailModel(**jload(weights_path))
        print("Loaded cached Pfail weights from:", weights_path)
    else:
        H_list, V_list, yfail_list = [], [], []
        print("Fitting logistic regression for Pfail on validation set...")

        for item in tqdm(val_items, desc="Validation (attempt0)", unit="task"):
            task_id = item["task_id"]
            raw_prompt = item["prompt"]
            test_src = item["test"]
            entry_point = item["entry_point"]

            cached = done.get(task_id, None)
            if cached and cached.get("attempt0") and cached["attempt0"].get("H") is not None:
                H = float(cached["attempt0"]["H"])
                V = float(cached["attempt0"]["V"])
                passed0 = bool(cached["attempt0"]["passed"])
            else:
                per_ent = []
                per_code = []
                for p_idx in range(CFG.n_prompts):
                    wrapped = apply_mte_prompt(raw_prompt, p_idx)
                    res = generate_with_entropy(
                        tok, model, wrapped, family=family,
                        max_new_tokens=CFG.max_new_tokens,
                        temperature=CFG.temp_attempt0, top_p=CFG.top_p_attempt0
                    )
                    per_ent.append(float(res["mean_token_entropy_nats"]))
                    per_code.append(res["generated_code"])

                H = float(np.mean(per_ent))
                V = float(np.var(per_ent))

                best_idx = int(np.argmin(per_ent))
                cand_code = per_code[best_idx]
                ok, _err = humaneval_pass(raw_prompt, cand_code, test_src, entry_point)
                passed0 = bool(ok)

                if task_id not in done:
                    done[task_id] = {
                        "task_id": task_id,
                        "model_id": model_id,
                        "family": family,
                        "size": size,
                    }
                done[task_id]["attempt0"] = {
                    "H": H, "V": V, "passed": passed0,
                    "selected_variant_idx": best_idx,
                    "selected_entropy": float(per_ent[best_idx]),
                }

            H_list.append(H)
            V_list.append(V)
            yfail_list.append(0 if passed0 else 1)

        H_arr = np.array(H_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        y_fail = np.array(yfail_list, dtype=int)

        pmodel = fit_pfail_model(H_arr, V_arr, y_fail)
        jdump(pmodel.__dict__, weights_path)
        print("Saved Pfail weights to:", weights_path)
        print("Weights:", {"w1": pmodel.w1, "w2": pmodel.w2, "b": pmodel.b})
-
    rng = np.random.default_rng(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    for item in tqdm(dataset, desc=f"Self-correcting eval ({model_id.split('/')[-1]})", unit="task"):
        task_id = item["task_id"]
        raw_prompt = item["prompt"]
        test_src = item["test"]
        entry_point = item["entry_point"]

        if CFG.use_cache and task_id in done and done[task_id].get("final") is not None:
            continue 

        attempt_records = []
        final_selected = None

        for attempt in range(CFG.max_attempts):
            temperature = CFG.temp_attempt0 if attempt == 0 else CFG.temp_regen
            top_p       = CFG.top_p_attempt0 if attempt == 0 else CFG.top_p_regen

            per_ent = []
            per_code = []
            for p_idx in range(CFG.n_prompts):
                wrapped = apply_mte_prompt(raw_prompt, p_idx)
                res = generate_with_entropy(
                    tok, model, wrapped, family=family,
                    max_new_tokens=CFG.max_new_tokens,
                    temperature=temperature, top_p=top_p
                )
                per_ent.append(float(res["mean_token_entropy_nats"]))
                per_code.append(res["generated_code"])

            H = float(np.mean(per_ent))
            V = float(np.var(per_ent))
            pf = pfail(pmodel, H, V)

            best_idx = int(np.argmin(per_ent))
            cand_code = per_code[best_idx]
            ok, err = humaneval_pass(raw_prompt, cand_code, test_src, entry_point)

            attempt_records.append({
                "attempt": attempt,
                "temperature": temperature,
                "top_p": top_p,
                "H": H,
                "V": V,
                "pfail": pf,
                "passed": bool(ok),
                "error": err,
                "selected_variant_idx": best_idx,
                "selected_entropy": float(per_ent[best_idx]),
                "code": cand_code,
            })

            if pf <= CFG.pfail_threshold:
                final_selected = attempt_records[-1]
                break

        if final_selected is None:
            final_selected = sorted(attempt_records, key=lambda r: r["pfail"])[0]

        out_row = {
            "task_id": task_id,
            "model_id": model_id,
            "family": family,
            "size": size,
            "pfail_threshold": CFG.pfail_threshold,
            "max_attempts": CFG.max_attempts,
            "n_prompts": CFG.n_prompts,
            "final": {
                k: final_selected[k] for k in [
                    "attempt", "H", "V", "pfail", "passed",
                    "selected_variant_idx", "selected_entropy"
                ]
            },
            "attempts": [
                {k: a[k] for k in [
                    "attempt","temperature","top_p","H","V","pfail","passed","selected_variant_idx","selected_entropy"
                ]}
                for a in attempt_records
            ],
            "final_code": final_selected["code"],
        }

        append_jsonl(tasklog_path, out_row)
        done[task_id] = out_row

    del model, tok
    torch.cuda.empty_cache()
    gc.collect()

    records = load_jsonl_as_list(tasklog_path)
    rows = []
    for r in records:
        f = r.get("final", {})
        rows.append({
            "task_id": r["task_id"],
            "model_id": r["model_id"],
            "family": r["family"],
            "size": r["size"],
            "attempt_used": f.get("attempt"),
            "H": f.get("H"),
            "V": f.get("V"),
            "pfail": f.get("pfail"),
            "passed": f.get("passed"),
            "selected_variant_idx": f.get("selected_variant_idx"),
            "selected_entropy": f.get("selected_entropy"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(df_csv_path, index=False)
    print("Saved:", df_csv_path)

    out_prefix = os.path.join(mdir, "plots")
    plot_pfail_hist_and_scatter(df, out_prefix)

    if df["passed"].notna().any():
        pr = float(df["passed"].mean())
        print("Pass rate (final):", pr, f"({int(df['passed'].sum())}/{len(df)})")
    return df

def main():
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    heval = load_dataset("openai_humaneval", split="test")
    total = len(heval)
    n = min(CFG.max_tasks, total)
    dataset = [heval[i] for i in range(n)]
    print(f"HumanEval tasks: using {n}/{total}")

    all_dfs = []
    for family, size, model_id in MODELS:
        df = evaluate_one_model(family, size, model_id, dataset)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(CFG.out_dir, "combined_results.csv")
    combined.to_csv(combined_path, index=False)
    print("Saved:", combined_path)

    print("\nPer-model final pass rates:")
    for mid in combined["model_id"].unique():
        md = combined[combined["model_id"] == mid]
        if len(md):
            print(mid, "->", float(md["passed"].mean()), f"({int(md['passed'].sum())}/{len(md)})")

if __name__ == "__main__":
    main()
