# ===============================================
# Colab script: Bayes Prompt Ensembles (BayesPE)
# for code-generation on HumanEval with Beta-Bernoulli
# ===============================================

import os, time, gc, math, json, random
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load as load_metric
import pandas as pd
import numpy as np
from scipy.special import softmax
from getpass import getpass

print("torch:", torch.__version__)

# ---------- HF login ----------
from huggingface_hub import login, whoami

HF_TOKEN = getpass("Paste your Hugging Face token: ").strip()
login(HF_TOKEN, add_to_git_credential=False)
print("Logged in as:", whoami().get("name", "unknown"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # needed by code_eval
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f"Using device={DEVICE}, dtype={DTYPE}")

# Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ===============================================
# Model loading + prompting
# ===============================================

def load_model(model_id: str, hf_token: str = None):
    """
    Simple model loader: GPU in bf16 if available, no quantization.
    """
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=hf_token,
        trust_remote_code=True
    )
    # Fallback pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",   # avoid flash-attn2 issues
    )
    model.eval()
    return tok, model

def build_prompt_for_model(raw_prompt: str, family: str) -> Dict[str, Any]:
    """
    Returns either:
      - {'chat': True, 'messages': [...]}
      - {'chat': False, 'text': "..."}
    So we can support chat-style (DeepSeek instruct) and plain text (Qwen-Coder).
    """
    if family in {"deepseek"}:
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

    # generic fallback
    return {"chat": False, "text": raw_prompt + "\n# Your code below:\n"}

# Prompt templates for Bayes Prompt Ensemble (these are the "a_i" prompts)
BAYESPE_TEMPLATES = [
    "You are an expert Python developer. Write only the function implementation.\n\n{p}\n# Your solution:\n",
    "Complete the Python function so it passes the tests. No explanations.\n\n{p}\n# Implementation:\n",
    "Return a minimal and correct Python solution for this function.\n\n{p}\n# Code:\n",
    "Implement the function described below in Python. Output only valid code.\n\n{p}\n# Solution:\n",
    "Write a concise Python implementation that satisfies the specification.\n\n{p}\n# Function:\n",
]

def apply_bayespe_prompt(raw_prompt: str, idx: int) -> str:
    """Pick one of the BayesPE templates and fill in the HumanEval prompt."""
    tpl = BAYESPE_TEMPLATES[idx % len(BAYESPE_TEMPLATES)]
    return tpl.format(p=raw_prompt.strip())

@torch.inference_mode()
def generate_for_one_prompt(tok, model, raw_prompt: str, family: str,
                            bayespe_index: int,
                            max_new_tokens: int = 320,
                            temperature: float = 0.2,
                            top_p: float = 0.95) -> str:
    """
    Generate a single code sample for a given BayesPE prompt index.
    We embed the HumanEval prompt in a natural-language template,
    then possibly in chat format depending on the family.
    """
    device = next(model.parameters()).device
    # Compose the BayesPE-wrapped prompt
    wrapped_text = apply_bayespe_prompt(raw_prompt, bayespe_index)
    # Now turn into model-specific prompt
    spec = build_prompt_for_model(wrapped_text, family)

    has_chat_template = getattr(tok, "chat_template", None) not in (None, "")

    if spec["chat"] and has_chat_template:
        # True chat model with template
        input_ids = tok.apply_chat_template(
            spec["messages"],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        attention_mask = None
    else:
        # Fallback: plain text prompt
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
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    out = model.generate(**gen_kwargs)
    gen_ids = out.sequences[0, input_ids.shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    return gen_text.strip()

# ===============================================
# Bayes Prompt Ensemble weight optimization
# ===============================================

def optimize_bayespe_weights(
    prompt_passes_val: List[List[bool]],
    alpha0: float = 1.0,
    beta0: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Beta–Bernoulli BayesPE-style weight learner.

    prompt_passes_val[t][i] = 1/0 indicating whether prompt i
    produced a passing solution on validation task t.

    For each prompt i we assume:
        θ_i ~ Beta(alpha0, beta0)
        y_{t,i} ~ Bernoulli(θ_i)

    Posterior for θ_i:
        θ_i | data ~ Beta(alpha0 + S_i, beta0 + F_i)

    We use the posterior mean E[θ_i | data] as a "skill" score
    and normalize into a distribution over prompts.
    """
    passes_arr = np.array(prompt_passes_val, dtype=np.float32)  # shape: (T, P)
    if passes_arr.ndim != 2:
        raise ValueError("prompt_passes_val must be [n_val_tasks][n_prompts].")

    n_tasks, n_prompts = passes_arr.shape
    successes = passes_arr.sum(axis=0)          # S_i
    failures  = n_tasks - successes             # F_i

    # Beta posterior parameters
    posterior_alpha = alpha0 + successes
    posterior_beta  = beta0 + failures

    # Posterior mean of θ_i for each prompt
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta + 1e-12)

    # Convert to weights with a softmax over log posterior mean
    logits = np.log(posterior_mean + 1e-12) / max(temperature, 1e-6)
    w = softmax(logits)

    return w

def predictive_uncertainty_from_pass_matrix(
    all_passes: List[List[bool]],
    weights: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Given all_passes[task][prompt] and prompt weights,
    compute per-task predictive success probs and entropies.

    For task j:
        p_success_j = sum_i w_i * y_{j,i}
        u_j = -[p log p + (1-p) log(1-p)]   (binary entropy)

    Returns:
        p_success (shape [T]), entropies (shape [T]), mean_entropy
    """
    passes_arr = np.array(all_passes, dtype=np.float32)  # (T, P)
    # mixture-of-experts success prob
    p_success = (passes_arr * weights[None, :]).sum(axis=1)
    eps = 1e-12
    ent = -(p_success * np.log(p_success + eps) +
            (1.0 - p_success) * np.log(1.0 - p_success + eps))
    mean_ent = float(np.mean(ent))
    return p_success, ent, mean_ent

# ===============================================
# HumanEval + code_eval setup
# ===============================================

heval = load_dataset("openai_humaneval")["test"]
print(f"HumanEval tasks total: {len(heval)}")
print("Example task_id:", heval[0]["task_id"])

code_eval = load_metric("code_eval")

# Config for experiments
MAX_TASKS_PER_MODEL = 40     # to keep runtime manageable; can increase to 164
N_PROMPTS_BPE        = 5     # number of BayesPE templates (<= len(BAYESPE_TEMPLATES))
N_VAL_TASKS          = 10    # number of tasks used to fit BayesPE weights (rest used for test)
MAX_NEW_TOKENS       = 320

# Models to evaluate (NO LLAMA here)
MODELS = [
    ("deepseek",   "≈1.3B-instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    ("qwen-coder", "≈1.5B",          "Qwen/Qwen2.5-Coder-1.5B"),
    ("qwen-coder", "≈3B",            "Qwen/Qwen2.5-Coder-3B"),
    # add more if VRAM/runtime allows
]

# ===============================================
# Main loop: for each model, run BayesPE + pass@k
# ===============================================

all_model_summaries = []

for family, size_bucket, model_id in MODELS:
    print("\n==============================")
    print(f"Model: {model_id} [{family} / {size_bucket}]")
    print("==============================")

    # Load model and tokenizer
    try:
        t0 = time.time()
        tok, model = load_model(model_id, hf_token=HF_TOKEN)
        print(f"Loaded in {time.time() - t0:.1f}s")
    except RuntimeError as e:
        print(f"Failed to load {model_id}: {e}")
        continue

    # Choose subset of HumanEval
    subset = [heval[i] for i in range(min(MAX_TASKS_PER_MODEL, len(heval)))]
    print(f"Using {len(subset)} HumanEval tasks for this model.")

    # Storage: per-task per-prompt candidates
    all_candidates = []           # candidates[task] = [code_from_prompt0, ..., prompt_{N-1}]
    all_passes     = []           # passes[task]    = [bool per prompt]
    all_tests      = []           # list of test strings
    all_task_ids   = []

    # Generate code for each task & each BayesPE prompt
    for idx, item in enumerate(subset):
        task_id = item["task_id"]
        raw_prompt = item["prompt"]
        test_code  = item["test"]

        per_prompt_codes = []

        print(f"  Task {idx+1}/{len(subset)} - {task_id}", flush=True)
        for p_idx in range(N_PROMPTS_BPE):
            try:
                code_str = generate_for_one_prompt(
                    tok, model,
                    raw_prompt=raw_prompt,
                    family=family,
                    bayespe_index=p_idx,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.2,
                    top_p=0.95
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM while generating for {model_id} / {task_id}. Skipping remaining prompts.")
                    code_str = ""
                else:
                    raise

            per_prompt_codes.append(code_str)

        # Evaluate all prompts for this task using code_eval
        tmp_results = code_eval.compute(
          references=[test_code],
          predictions=[per_prompt_codes],
          k=[len(per_prompt_codes)],
          num_workers=4,
          timeout=10.0
        )

        # New structure: tmp_results is already a list of dicts with "passed"
        if isinstance(tmp_results, list) and len(tmp_results) > 0 and "passed" in tmp_results[0]:
            pass_list = tmp_results[0]["passed"]
        else:
            # fallback: treat everything as fail
            pass_list = [False] * len(per_prompt_codes)

        all_candidates.append(per_prompt_codes)
        all_passes.append(pass_list)
        all_tests.append(test_code)
        all_task_ids.append(task_id)

    # --------------------------
    # Split into validation and test for BayesPE
    # with shuffling so validation comes from
    # different parts of the dataset.
    # --------------------------
    n_total = len(all_candidates)
    n_val   = min(N_VAL_TASKS, n_total)

    task_perm = list(range(n_total))
    random.shuffle(task_perm)

    val_indices  = task_perm[:n_val]
    test_indices = task_perm[n_val:]

    passes_val = [all_passes[i] for i in val_indices]

    if n_val > 0:
        bayespe_weights = optimize_bayespe_weights(passes_val)
    else:
        bayespe_weights = np.ones(N_PROMPTS_BPE) / N_PROMPTS_BPE

    print("\nBayesPE weights over prompts (Beta–Bernoulli):")
    for i, w in enumerate(bayespe_weights):
        print(f"  Prompt {i}: w={w:.3f}")

    # Per-task BayesPE predictive uncertainty
    _, task_entropies, mean_entropy = predictive_uncertainty_from_pass_matrix(
        all_passes,
        bayespe_weights
    )
    print(f"Mean BayesPE predictive entropy across tasks: {mean_entropy:.4f}")

    # Ensemble metrics: pass@1..K using all candidates
    k_values = [1, min(3, N_PROMPTS_BPE), N_PROMPTS_BPE]
    metrics_ensemble, results_ensemble = code_eval.compute(
        references=all_tests,
        predictions=all_candidates,
        k=k_values,
        num_workers=4,
        timeout=10.0
    )

    # BayesPE "best prompt" pass@1: pick prompt index with max weight
    best_prompt_idx = int(np.argmax(bayespe_weights))
    bpe_best_codes = []
    for codes in all_candidates:
        if best_prompt_idx < len(codes):
            bpe_best_codes.append(codes[best_prompt_idx])
        else:
            bpe_best_codes.append(codes[0])

    metrics_bpe_best, _ = code_eval.compute(
        references=all_tests,
        predictions=[[c] for c in bpe_best_codes],
        k=[1],
        num_workers=4,
        timeout=10.0
    )
    pass1_bpe = metrics_bpe_best["pass@1"]

    # Print summary for this model
    print("\nSummary for model:", model_id)
    for k in k_values:
        print(f"  Ensemble pass@{k}: {metrics_ensemble.get(f'pass@{k}', float('nan')):.3f}")
    print(f"  BayesPE best-prompt pass@1: {pass1_bpe:.3f}")
    print(f"  Mean BayesPE predictive entropy: {mean_entropy:.4f}")

    # Save per-model summary
    all_model_summaries.append({
        "family": family,
        "size_bucket": size_bucket,
        "model_id": model_id,
        "n_tasks": n_total,
        "n_prompts": N_PROMPTS_BPE,
        "bayespe_weights": bayespe_weights.tolist(),
        "ensemble_metrics": {k: float(metrics_ensemble.get(f"pass@{k}", float("nan"))) for k in k_values},
        "bayespe_pass1": float(pass1_bpe),
        "mean_bayespe_entropy": float(mean_entropy),
    })

    # Free memory before next model
    del model; del tok
    torch.cuda.empty_cache(); gc.collect()

# ===============================================
# Final comparison summary + simple correlation
# ===============================================

print("\n================ FINAL COMPARISON ================")
for s in all_model_summaries:
    print(f"\nModel: {s['model_id']} [{s['family']}, {s['size_bucket']}]")
    print(f"  Tasks: {s['n_tasks']} | #Prompts: {s['n_prompts']}")
    for k, v in s["ensemble_metrics"].items():
        print(f"  Ensemble pass@{k}: {v:.3f}")
    print(f"  BayesPE best-prompt pass@1: {s['bayespe_pass1']:.3f}")
    print(f"  Mean BayesPE predictive entropy: {s['mean_bayespe_entropy']:.4f}")
    print("  Weights:", ["{:.2f}".format(w) for w in s["bayespe_weights"]])

# Save per-model summary
summary_df = pd.DataFrame(all_model_summaries)
summary_df.to_csv("bayes_prompt_ensemble_summary_no_llama_beta_bernoulli.csv", index=False)
print("\nSaved: bayes_prompt_ensemble_summary_no_llama_beta_bernoulli.csv")

# Correlation across models: mean BayesPE entropy vs ensemble pass@1
if len(all_model_summaries) >= 2:
    entropies = np.array([m["mean_bayespe_entropy"] for m in all_model_summaries], dtype=float)
    pass1s    = np.array([m["ensemble_metrics"][1] for m in all_model_summaries], dtype=float)

    if np.all(np.isfinite(entropies)) and np.all(np.isfinite(pass1s)):
        pearson = np.corrcoef(entropies, pass1s)[0, 1]
        print(f"\nCorrelation(mean BayesPE entropy, ensemble pass@1) = {pearson:.4f}")
    else:
        print("\nCould not compute correlation: non-finite values present.")
else:
    print("\nNot enough models to compute a meaningful correlation.")

# ===============================================
# Scatter plot: Uncertainty vs pass@1
# ===============================================

import matplotlib.pyplot as plt

model_names = [m["model_id"] for m in all_model_summaries]
uncertainties = np.array([m["mean_bayespe_entropy"] for m in all_model_summaries], dtype=float)
pass1_scores = np.array([m["ensemble_metrics"][1] for m in all_model_summaries], dtype=float)

plt.figure(figsize=(8,6))
plt.scatter(uncertainties, pass1_scores, s=120, c='royalblue')

# Add text labels for models
for name, x, y in zip(model_names, uncertainties, pass1_scores):
    plt.text(x + 0.0005, y + 0.0005, name, fontsize=9)

plt.xlabel("Mean BayesPE Predictive Entropy (Uncertainty)", fontsize=12)
plt.ylabel("Ensemble pass@1", fontsize=12)
plt.title("Model Uncertainty vs pass@1 Performance", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
