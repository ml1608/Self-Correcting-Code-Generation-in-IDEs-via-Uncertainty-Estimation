# ============================================================
# Mean Token Entropy + Logistic Pfail on HumanEval
# Small models: DeepSeek 1.3B-instruct, Qwen2.5-Coder-1.5B
# Uses temperature = 0.6, top_p = 0.92
# Produces bar charts: mean Pfail vs pass@1 per model
# ============================================================

import os, time, gc, random
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load as load_metric
import pandas as pd
import numpy as np
from getpass import getpass

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from huggingface_hub import login, whoami

print("torch:", torch.__version__)

# ============================================================
# Hugging Face login
# ============================================================

HF_TOKEN = getpass("Paste your Hugging Face token: ").strip()
login(HF_TOKEN, add_to_git_credential=False)
print("Logged in as:", whoami().get("name", "unknown"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f"Using device={DEVICE}, dtype={DTYPE}")

# Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ============================================================
# CONFIG
# ============================================================

MAX_TASKS_PER_MODEL = 20      # you can increase later (40, 80, 164)
N_PROMPTS_MTE        = 5      # prompts in ensemble
N_VAL_TASKS          = 8      # tasks to train logistic regression
MAX_NEW_TOKENS       = 200

GEN_TEMPERATURE = 0.6
GEN_TOP_P       = 0.92

# Just the two small models you care about
MODELS = [
    ("deepseek",   "≈1.3B-instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    ("qwen-coder", "≈1.5B",          "Qwen/Qwen2.5-Coder-1.5B"),
]

# ============================================================
# Model loading + prompting
# ============================================================

def load_model(model_id: str, hf_token: str = None):
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=hf_token,
        trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model

def build_prompt_for_model(raw_prompt: str, family: str) -> Dict[str, Any]:
    # Same structure as your friend’s code
    if family in {"llama", "deepseek"}:
        system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
        user = raw_prompt + "\n\n# Your code below:\n"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return {"chat": True, "messages": messages}

    if family == "qwen-coder":
        text = raw_prompt + "\n# Your code below:\n"
        return {"chat": False, "text": text}

    return {"chat": False, "text": raw_prompt + "\n# Your code below:\n"}

BAYESPE_TEMPLATES = [
    "You are an expert Python developer. Write only the function implementation.\n\n{p}\n# Your solution:\n",
    "Complete the Python function so it passes the tests. No explanations.\n\n{p}\n# Implementation:\n",
    "Return a minimal and correct Python solution for this function.\n\n{p}\n# Code:\n",
    "Implement the function described below in Python. Output only valid code.\n\n{p}\n# Solution:\n",
    "Write a concise Python implementation that satisfies the specification.\n\n{p}\n# Function:\n",
]

def apply_mte_prompt(raw_prompt: str, idx: int) -> str:
    tpl = BAYESPE_TEMPLATES[idx % len(BAYESPE_TEMPLATES)]
    return tpl.format(p=raw_prompt.strip())

def token_entropies_from_logits_list(scores: List[torch.Tensor]) -> Tuple[List[float], float]:
    H = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        H.append(ent)
    return H, (float(np.mean(H)) if H else float("nan"))

@torch.inference_mode()
def generate_with_entropy(
    tok,
    model,
    raw_prompt: str,
    family: str,
    max_new_tokens: int = 320,
    temperature: float = 0.6,
    top_p: float = 0.92,
) -> Dict[str, Any]:
    """
    Stochastic generation (temperature, top_p) + per-step logits -> mean token entropy.
    This is basically your friend's generate_with_entropy, but with
    temperature=0.6, top_p=0.92 by default.
    """
    device = next(model.parameters()).device
    # Wrap the prompt with one of the templates outside this function
    spec = build_prompt_for_model(raw_prompt, family)

    has_chat_template = getattr(tok, "chat_template", None) not in (None, "")

    if spec["chat"] and has_chat_template:
        input_ids = tok.apply_chat_template(
            spec["messages"], add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        attention_mask = None
    else:
        if spec["chat"]:
            parts = []
            for msg in spec["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"[SYSTEM] {content}\n")
                elif role == "user":
                    parts.append(f"[USER] {content}\n")
                else:
                    parts.append(f"[{role.upper()}] {content}\n")
            text = "".join(parts) + "\n[ASSISTANT]\n"
        else:
            text = spec["text"]

        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,                  # because temperature=0.6
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
        "generated_code": gen_text,
        "mean_token_entropy_nats": H_mean,
        "first5_entropies": H_list[:5],
        "num_steps": len(H_list),
        "num_chars": len(gen_text),
    }

# ============================================================
# HumanEval + code_eval
# ============================================================

heval = load_dataset("openai_humaneval")["test"]
print(f"HumanEval tasks total: {len(heval)}")
print("Example task_id:", heval[0]["task_id"])

code_eval = load_metric("code_eval")

# ============================================================
# Main: MTE features (H, V) + logistic Pfail + pass@1
# ============================================================

all_model_summaries: List[Dict[str, Any]] = []

for family, size_bucket, model_id in MODELS:
    print("\n==============================")
    print(f"Model: {model_id} [{family} / {size_bucket}]")
    print("==============================")

    try:
        t0 = time.time()
        tok, model = load_model(model_id, hf_token=HF_TOKEN)
        print(f"Loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        continue

    subset = [heval[i] for i in range(min(MAX_TASKS_PER_MODEL, len(heval)))]
    print(f"Using {len(subset)} HumanEval tasks for this model.")

    per_task_mean_entropies: List[float] = []   # H(x_j)
    per_task_var_entropies:  List[float] = []   # V(x_j)
    per_task_fail_label:     List[int]   = []   # y_j  (1 = fail, 0 = success)

    for idx, item in enumerate(subset):
        task_id = item["task_id"]
        raw_prompt = item["prompt"]
        test_code  = item["test"]

        per_prompt_codes = []
        per_prompt_mean_entropies = []

        print(f"  Task {idx+1}/{len(subset)} - {task_id}", flush=True)

        # Generate once per prompt template (with templates applied)
        for p_idx in range(N_PROMPTS_MTE):
            wrapped_prompt = apply_mte_prompt(raw_prompt, p_idx)
            try:
                res = generate_with_entropy(
                    tok,
                    model,
                    wrapped_prompt,      # note: wrapped here
                    family=family,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM on {model_id} / {task_id}. Remaining prompts empty.")
                    res = {
                        "generated_code": "",
                        "mean_token_entropy_nats": 0.0,
                        "first5_entropies": [],
                        "num_steps": 0,
                        "num_chars": 0,
                    }
                else:
                    raise

            per_prompt_codes.append(res["generated_code"])
            per_prompt_mean_entropies.append(res["mean_token_entropy_nats"])

        # Evaluate all prompts for this task using code_eval to get pass/fail
        tmp_results = code_eval.compute(
          references=[test_code],
          predictions=[per_prompt_codes],
          k=[len(per_prompt_codes)],
          num_workers=4,
          timeout=10.0
        )

        # Robust pass_list extraction (code_eval versions differ)
        if isinstance(tmp_results, list):
            d0 = tmp_results[0]
            if "passed" in d0:
                pass_list = d0["passed"]
            elif "results" in d0 and len(d0["results"]) > 0 and "passed" in d0["results"][0]:
                pass_list = d0["results"][0]["passed"]
            else:
                pass_list = [False] * len(per_prompt_codes)
        elif isinstance(tmp_results, dict):
            if "passed" in tmp_results:
                passed = tmp_results["passed"]
                pass_list = passed[0] if isinstance(passed[0], list) else passed
            elif "results" in tmp_results and "passed" in tmp_results["results"][0]:
                pass_list = tmp_results["results"][0]["passed"]
            else:
                pass_list = [False] * len(per_prompt_codes)
        else:
            pass_list = [False] * len(per_prompt_codes)

        pass_arr = np.array(pass_list, dtype=bool)
        task_success = bool(pass_arr.any())
        y_j = 1 if not task_success else 0  # 1 = fail, 0 = success

        H_j = float(np.mean(per_prompt_mean_entropies))
        V_j = float(np.var(per_prompt_mean_entropies))

        per_task_mean_entropies.append(H_j)
        per_task_var_entropies.append(V_j)
        per_task_fail_label.append(y_j)

    per_task_mean_entropies = np.array(per_task_mean_entropies, dtype=float)
    per_task_var_entropies  = np.array(per_task_var_entropies, dtype=float)
    per_task_fail_label     = np.array(per_task_fail_label, dtype=int)
    n_tasks = len(per_task_fail_label)

    # ========================================================
    # Train logistic regression on validation subset
    # ========================================================
    n_val = min(N_VAL_TASKS, n_tasks)
    perm = np.arange(n_tasks)
    np.random.shuffle(perm)
    val_idx = perm[:n_val]

    H_val = per_task_mean_entropies[val_idx]
    V_val = per_task_var_entropies[val_idx]
    y_val = per_task_fail_label[val_idx]

    # Standardize H & V
    H_mean, H_std = float(H_val.mean()), float(H_val.std() if H_val.std() > 1e-8 else 1.0)
    V_mean, V_std = float(V_val.mean()), float(V_val.std() if V_val.std() > 1e-8 else 1.0)

    H_val_std = (H_val - H_mean) / H_std
    V_val_std = (V_val - V_mean) / V_std
    X_val = np.stack([H_val_std, V_val_std], axis=1)

    # Fit logistic regression y ~ H,V  (predict Pfail)
    if len(np.unique(y_val)) > 1:
        lr = LogisticRegression()
        lr.fit(X_val, y_val)
        w1, w2 = lr.coef_[0]
        b = lr.intercept_[0]
    else:
        # Degenerate case: all success or all fail in validation
        p = float(y_val.mean()) if y_val.size > 0 else 0.5
        p = min(max(p, 1e-4), 1 - 1e-4)
        w1 = 0.0
        w2 = 0.0
        b  = np.log(p / (1 - p))

    print(f"\nLearned logistic weights for Pfail:")
    print(f"  w1 (H): {w1:.4f}, w2 (V): {w2:.4f}, b: {b:.4f}")

    # ========================================================
    # Inference: Pfail(x) for ALL tasks
    # ========================================================
    H_all_std = (per_task_mean_entropies - H_mean) / H_std
    V_all_std = (per_task_var_entropies  - V_mean) / V_std
    logits = w1 * H_all_std + w2 * V_all_std + b
    Pfail_all = 1.0 / (1.0 + np.exp(-logits))

    mean_pfail = float(Pfail_all.mean())
    std_pfail  = float(Pfail_all.std())
    print(f"Mean Pfail (uncertainty): {mean_pfail:.4f} ± {std_pfail:.4f}")

    pass_at_1 = float((per_task_fail_label == 0).mean())
    print(f"pass@1 (any-prompt success rate): {pass_at_1:.3f}")

    all_model_summaries.append({
        "family": family,
        "size_bucket": size_bucket,
        "model_id": model_id,
        "n_tasks": n_tasks,
        "n_prompts": N_PROMPTS_MTE,
        "mean_H": float(per_task_mean_entropies.mean()),
        "std_H":  float(per_task_mean_entropies.std()),
        "mean_V": float(per_task_var_entropies.mean()),
        "std_V":  float(per_task_var_entropies.std()),
        "w1": float(w1),
        "w2": float(w2),
        "b":  float(b),
        "mean_pfail": mean_pfail,
        "std_pfail":  std_pfail,
        "pass_at_1":  pass_at_1,
    })

    del model; del tok
    torch.cuda.empty_cache(); gc.collect()

# ============================================================
# Save + print summary
# ============================================================

summary_df = pd.DataFrame(all_model_summaries)
summary_df.to_csv("mte_logistic_uncertainty_pass1_smallmodels_temp0.6_top0.92.csv", index=False)
print("\nSaved: mte_logistic_uncertainty_pass1_smallmodels_temp0.6_top0.92.csv")

print("\n================ FINAL SUMMARY ================")
for s in all_model_summaries:
    print(f"\nModel: {s['model_id']} [{s['family']}, {s['size_bucket']}]")
    print(f"  Tasks: {s['n_tasks']} | #Prompts: {s['n_prompts']}")
    print(f"  mean_H: {s['mean_H']:.4f} ± {s['std_H']:.4f}")
    print(f"  mean_V: {s['mean_V']:.4f} ± {s['std_V']:.4f}")
    print(f"  mean_Pfail: {s['mean_pfail']:.4f} ± {s['std_pfail']:.4f}")
    print(f"  pass@1: {s['pass_at_1']:.3f}")
    print(f"  w1: {s['w1']:.4f}, w2: {s['w2']:.4f}, b: {s['b']:.4f}")

# ============================================================
# BAR CHARTS: Pfail vs pass@1 (per model)
# ============================================================

sns.set(style="whitegrid", font_scale=1.1)

plot_df = summary_df.sort_values(["family", "size_bucket"]).reset_index(drop=True)
x = np.arange(len(plot_df))

palette = {
    "deepseek":   "#2b4c7e",
    "qwen-coder": "#4caf50",
}
colors = plot_df["family"].map(palette)

# Chart 1: Mean Pfail + error bars
plt.figure(figsize=(8, 5))
plt.bar(
    x,
    plot_df["mean_pfail"],
    yerr=plot_df["std_pfail"],
    capsize=4,
    color=colors,
    edgecolor="black",
)
plt.xticks(x, plot_df["model_id"], rotation=45, ha="right")
plt.ylabel("Mean Pfail (uncertainty)")
plt.xlabel("Model")
plt.ylim(0.0, 1.0)
plt.title("Mean Uncertainty (Pfail) – MTE + Logistic")
handles = [plt.Rectangle((0,0),1,1,color=palette[f]) for f in palette]
labels  = list(palette.keys())
plt.legend(handles, labels, title="Family")
plt.tight_layout()
plt.show()

# Chart 2: pass@1
plt.figure(figsize=(8, 5))
plt.bar(
    x,
    plot_df["pass_at_1"],
    color=colors,
    edgecolor="black",
)
plt.xticks(x, plot_df["model_id"], rotation=45, ha="right")
plt.ylabel("Pass@1 (fraction of tasks)")
plt.xlabel("Model")
plt.ylim(0.0, 1.0)
plt.title("Pass@1 – Small Models on HumanEval")
handles = [plt.Rectangle((0,0),1,1,color=palette[f]) for f in palette]
labels  = list(palette.keys())
plt.legend(handles, labels, title="Family")
plt.tight_layout()
plt.show()

# Chart 3: dual-axis bar chart Pfail vs pass@1
fig, ax1 = plt.subplots(figsize=(8, 5))

bar_width = 0.35
ax1.set_xlabel("Model")
ax1.set_ylabel("Mean Pfail", color="tab:red")
bars1 = ax1.bar(
    x - bar_width/2,
    plot_df["mean_pfail"],
    width=bar_width,
    color="tab:red",
    alpha=0.7,
    label="Mean Pfail",
)
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.set_ylim(0.0, 1.0)

ax2 = ax1.twinx()
ax2.set_ylabel("Pass@1", color="tab:blue")
bars2 = ax2.bar(
    x + bar_width/2,
    plot_df["pass_at_1"],
    width=bar_width,
    color="tab:blue",
    alpha=0.7,
    label="Pass@1",
)
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2.set_ylim(0.0, 1.0)

plt.xticks(x, plot_df["model_id"], rotation=45, ha="right")
ax1.legend([bars1, bars2], ["Mean Pfail", "Pass@1"], loc="upper left")
plt.title("Uncertainty (Pfail) vs Pass@1 by Model (MTE Ensembles)")
plt.tight_layout()
plt.show() 