import os, re, gc, signal
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


# 0) HF Auth (needed for gated models like Llama)

if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Set HF_TOKEN in os.environ (Colab secrets) for gated model access.")
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

# 1) CONFIG (tuned for stability)

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

NUM_EXAMPLES = 100
K_PASS = 3
L_SELF = 2

MAX_NEW_TOKENS_CODE = 512
GEN_TIMEOUT_S = 120

ROLLOUT_TEMP = 0.6
ROLLOUT_TOP_P = 0.95

SELF_EVAL_TEMP = 0.7
SELF_EVAL_MAX_NEW_TOKENS = 5
SELF_EVAL_TIMEOUT_S = 30

# Regeneration experiment params
TAU = 0.3             # trigger threshold on verifier P(True)
K_REGEN = 2           # rollouts after regeneration to estimate new P(True)
REGEN_TEMP = 0.7
REGEN_TOP_P = 0.9
REGEN_TIMEOUT_S = 120

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SYSTEM_PROMPT = (
    "You are a Python coding assistant. "
    "Complete the function so that it passes the tests. "
    "Return only Python code, no explanation."
)

# 2) Timeout util 

class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# 3) HumanEval dataset
raw = load_dataset("openai_humaneval", split="test")

# 4) Parser
def parse_code(text: str) -> str:
    if text is None:
        return ""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", str(text), re.DOTALL | re.IGNORECASE)
    code = blocks[-1].strip() if blocks else str(text).strip()
    if "def " in code:
        code = code[code.find("def "):].strip()
    try:
        compile(code, "<candidate>", "exec")
    except Exception:
        return ""
    return code


# 5) Execution verifier (with timeout)
def passes_humaneval_tests(prompt, completion_text, info, timeout_s: int = 6):
    code = parse_code(completion_text)
    if not code:
        return 0.0
    module_src = prompt + "\n" + code + "\n\n" + info["test"]
    glb = {}
    try:
        with time_limit(timeout_s):
            exec(module_src, glb, glb)
            fn = glb[info["entry_point"]]
            glb["candidate"] = fn
            glb["check"](fn)
        return 1.0
    except Exception:
        return 0.0

# 6) Agreement score
def canonicalize_code(code: str) -> str:
    if not code:
        return ""
    lines = [ln.rstrip() for ln in code.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    return "\n".join(lines).strip()

def agreement_ptrue(codes):
    canon = [canonicalize_code(c) for c in codes if canonicalize_code(c)]
    if not canon:
        return 0.0
    counts = {}
    for c in canon:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.values()) / len(codes)

# 7) Baseline self-eval
def parse_true_false(text: str):
    t = (text or "").strip().lower()
    if t.startswith("true"): return 1
    if t.startswith("false"): return 0
    if re.search(r"\btrue\b", t) and not re.search(r"\bfalse\b", t): return 1
    if re.search(r"\bfalse\b", t) and not re.search(r"\btrue\b", t): return 0
    return None

def self_eval_prompt(problem_prompt: str, candidate_code: str):
    return (
        "You are a strict evaluator for Python coding tasks.\n"
        "Answer with exactly one word: True or False.\n\n"
        "Problem:\n"
        f"{problem_prompt}\n\n"
        "Candidate solution:\n"
        "```python\n"
        f"{candidate_code}\n"
        "```\n\n"
        "Question: Does the candidate solution pass all tests?\n"
        "Answer:"
    )


# 8) Local generation
def build_chat_prompt(tokenizer, system, user):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        msgs = [{"role":"system","content":system},{"role":"user","content":user}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"

def generate_text(model, tokenizer, system_prompt, user_prompt,
                  max_new_tokens, temperature=0.0, top_p=1.0):
    prompt = build_chat_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

def safe_generate(model, tokenizer, system_prompt, user_prompt,
                  max_new_tokens, temperature=0.0, top_p=1.0,
                  timeout_s=120):
    try:
        with time_limit(timeout_s):
            return generate_text(
                model, tokenizer, system_prompt, user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
    except Exception:
        return ""


# 9) Metrics
def compute_metrics(df: pd.DataFrame, score_col: str):
    sub = df[["y_correct", score_col]].dropna()
    y = sub["y_correct"].astype(int).values
    s = sub[score_col].astype(float).values

    auroc = float("nan")
    if len(np.unique(y)) == 2:
        try:
            auroc = roc_auc_score(y, s)
        except Exception:
            auroc = float("nan")

    rho, pval = float("nan"), float("nan")
    if len(s) >= 2 and np.std(s) > 0 and len(np.unique(y)) == 2:
        try:
            rho, pval = spearmanr(s, y)
        except Exception:
            rho, pval = float("nan"), float("nan")

    return {
        "n": int(len(sub)),
        "pos": int(y.sum()),
        "neg": int(len(y) - y.sum()),
        "AUROC": auroc,
        "Spearman_rho": rho,
        "Spearman_p": pval,
    }

# 10) Failure cases
def get_failure_cases(df: pd.DataFrame, n=3):
    # Failure A: overconfidence (baseline failure)
    fail_overconf = df[(df["y_correct"] == 0) & (df["p_true_self"] >= 0.7)].head(n)

    # Failure B: correct uncertainty (verifier success)
    fail_low_ver = df[(df["y_correct"] == 0) & (df["p_true_verifier"] <= 0.2)].head(n)
    return fail_overconf, fail_low_ver

# 11) Regeneration experiment 
def regenerate_once(model, tokenizer, prompt):
    # "previous attempt failed" nudge
    user = prompt + "\n\nThe previous attempt failed. Try a different approach."
    return safe_generate(
        model, tokenizer, SYSTEM_PROMPT, user,
        max_new_tokens=MAX_NEW_TOKENS_CODE,
        temperature=REGEN_TEMP,
        top_p=REGEN_TOP_P,
        timeout_s=REGEN_TIMEOUT_S
    )

def estimate_ptrue_from_rollouts(model, tokenizer, prompt, info, K: int):
    flags = []
    for _ in range(K):
        txt = safe_generate(
            model, tokenizer, SYSTEM_PROMPT, prompt,
            max_new_tokens=MAX_NEW_TOKENS_CODE,
            temperature=ROLLOUT_TEMP,
            top_p=ROLLOUT_TOP_P,
            timeout_s=GEN_TIMEOUT_S
        )
        flags.append(int(passes_humaneval_tests(prompt, txt, info, timeout_s=6) == 1.0))
    return float(np.mean(flags))

# 12) Plots
def plot_panels(df, score_col, title_prefix, save_path=None):
    correct = df[df["y_correct"] == 1][score_col].values
    incorrect = df[df["y_correct"] == 0][score_col].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)

    bins = np.linspace(
        min(df[score_col].min(), 0),
        df[score_col].max(),
        30
    )

    axes[0].hist(
        correct,
        bins=bins,
        color="royalblue",
        alpha=0.7,
        label=f"Correct (n={len(correct)})"
    )
    axes[0].hist(
        incorrect,
        bins=bins,
        color="red",
        alpha=0.6,
        label=f"Incorrect (n={len(incorrect)})"
    )

    axes[0].set_title(f"Histogram of Uncertainty: {title_prefix}", fontsize=12)
    axes[0].set_xlabel("Uncertainty / Confidence", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        correct,
        np.ones_like(correct),
        color="royalblue",
        alpha=0.7,
        label="Correct"
    )
    axes[1].scatter(
        incorrect,
        np.zeros_like(incorrect),
        color="red",
        alpha=0.7,
        label="Incorrect"
    )

    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Incorrect", "Correct"])
    axes[1].set_ylim(-0.2, 1.2)

    axes[1].set_title(f"Uncertainty vs Correctness: {title_prefix}", fontsize=12)
    axes[1].set_xlabel("Uncertainty / Confidence", fontsize=11)
    axes[1].set_ylabel("Correctness (0 = Incorrect, 1 = Correct)", fontsize=11)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved:", save_path)
    plt.show()


# 13) Main per model
all_results = []
summary_rows = []

for model_id in MODELS:
    print("\n==============================")
    print("Model:", model_id)
    print("==============================")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"], use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=os.environ["HF_TOKEN"],
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        model.eval()
    except Exception as e:
        print("LOAD FAILED:", repr(e))
        continue

    rows = []
    subset = raw.select(range(NUM_EXAMPLES))

    for i, row in enumerate(subset):
        prompt = row["prompt"]
        info = {"test": row["test"], "entry_point": row["entry_point"], "task_id": row["task_id"]}

        # --- Greedy for y_correct ---
        greedy_text = safe_generate(
            model, tokenizer, SYSTEM_PROMPT, prompt,
            max_new_tokens=MAX_NEW_TOKENS_CODE,
            temperature=0.0, top_p=1.0,
            timeout_s=GEN_TIMEOUT_S
        )
        greedy_code = parse_code(greedy_text)
        y_correct = int(passes_humaneval_tests(prompt, greedy_text, info, timeout_s=6) == 1.0) if greedy_code else 0

        # --- K rollouts for verifier P(True) ---
        pass_flags = []
        rollout_codes = []
        for _ in range(K_PASS):
            samp_text = safe_generate(
                model, tokenizer, SYSTEM_PROMPT, prompt,
                max_new_tokens=MAX_NEW_TOKENS_CODE,
                temperature=ROLLOUT_TEMP, top_p=ROLLOUT_TOP_P,
                timeout_s=GEN_TIMEOUT_S
            )
            rollout_codes.append(parse_code(samp_text))
            pass_flags.append(int(passes_humaneval_tests(prompt, samp_text, info, timeout_s=6) == 1.0))

        p_true_verifier = float(np.mean(pass_flags))
        p_true_agree = float(agreement_ptrue(rollout_codes))

        # --- Baseline self-eval on greedy code ---
        if greedy_code:
            votes = []
            se_prompt = self_eval_prompt(prompt, greedy_code)
            for _ in range(L_SELF):
                se_text = safe_generate(
                    model, tokenizer, "Answer strictly True or False.", se_prompt,
                    max_new_tokens=SELF_EVAL_MAX_NEW_TOKENS,
                    temperature=SELF_EVAL_TEMP, top_p=1.0,
                    timeout_s=SELF_EVAL_TIMEOUT_S
                )
                v = parse_true_false(se_text)
                votes.append(0 if v is None else v)
            p_true_self = float(np.mean(votes))
        else:
            p_true_self = 0.5

        rows.append({
            "task_id": info["task_id"],
            "y_correct": y_correct,
            "p_true_verifier": p_true_verifier,
            "p_true_agree": p_true_agree,
            "p_true_self": p_true_self,
        })

        if (i + 1) % 5 == 0:
            print(f"  done {i+1}/{NUM_EXAMPLES}")

    df = pd.DataFrame(rows)
    safe = model_id.replace("/", "_").replace(":", "_")

    # Save per-model CSV
    df.to_csv(f"{safe}_ptrue_compare.csv", index=False)
    print("Saved:", f"{safe}_ptrue_compare.csv")

    # Metrics
    m_self = compute_metrics(df, "p_true_self")
    m_ver  = compute_metrics(df, "p_true_verifier")
    m_ag   = compute_metrics(df, "p_true_agree")

    summary_rows.append({
        "model": model_id,
        "pos": m_ver["pos"],
        "neg": m_ver["neg"],
        "AUROC_self": m_self["AUROC"],
        "AUROC_verifier": m_ver["AUROC"],
        "AUROC_agree": m_ag["AUROC"],
        "Spearman_self": m_self["Spearman_rho"],
        "Spearman_verifier": m_ver["Spearman_rho"],
        "Spearman_agree": m_ag["Spearman_rho"],
    })

    # Failure cases (top 3)
    failA, failB = get_failure_cases(df, n=3)
    print("\nFailure A (incorrect + high baseline confidence):")
    display(failA)
    print("\nFailure B (incorrect + low verifier confidence):")
    display(failB)

    # --- Regeneration experiment ---
    delta_rows = []
    for idx, r in df.iterrows():
        if r["p_true_verifier"] < TAU:
            prompt = raw[idx]["prompt"]
            info = {"test": raw[idx]["test"], "entry_point": raw[idx]["entry_point"], "task_id": raw[idx]["task_id"]}

            # regenerate once (new attempt)
            new_text = regenerate_once(model, tokenizer, prompt)

            # estimate new P(True) with K_REGEN rollouts (cheap but meaningful)
            new_p = estimate_ptrue_from_rollouts(model, tokenizer, prompt, info, K=K_REGEN)

            delta_rows.append(new_p - r["p_true_verifier"])

    if len(delta_rows) == 0:
        print("\nNo examples triggered regeneration (all p_true_verifier >= TAU).")
    else:
        print("\nMean ΔP(True) after regeneration:", float(np.mean(delta_rows)))

        plt.figure(figsize=(7,4), dpi=150)
        plt.hist(delta_rows, bins=20, alpha=0.8)
        plt.axvline(0, color="red", linestyle="--")
        plt.title(f"{model_id} — Change in P(True) after regeneration")
        plt.xlabel("ΔP(True)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{safe}_delta_ptrue.png", bbox_inches="tight")
        plt.show()

    # Plots
    plot_panels(df, "p_true_verifier", f"{model_id} — Verifier P(True) (pass@{K_PASS}/{K_PASS})",
                save_path=f"{safe}_verifier_ptrue.png")
    plot_panels(df, "p_true_self", f"{model_id} — Baseline Self-eval P(True)",
                save_path=f"{safe}_self_ptrue.png")
    plot_panels(df, "p_true_agree", f"{model_id} — Agreement P(True) (mode/K)",
                save_path=f"{safe}_agree_ptrue.png")

    all_results.append(df.assign(model=model_id))

    # Free RAM between models
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# Save combined CSV + summary
if all_results:
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv("ALL_MODELS_ptrue_compare.csv", index=False)
    print("\nSaved: ALL_MODELS_ptrue_compare.csv")
    display(df_all.head())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("SUMMARY_metrics.csv", index=False)
    print("\nSaved: SUMMARY_metrics.csv")
    display(summary)
else:
    print("\nNo models produced results.")
