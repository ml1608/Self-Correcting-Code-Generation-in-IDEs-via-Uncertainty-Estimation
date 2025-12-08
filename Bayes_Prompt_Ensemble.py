# ============================================================
# Mean Token Entropy + Logistic Uncertainty on HumanEval
# Models: LLaMA 3 3B, Qwen2.5-Coder 3B, DeepSeek R1 3B
# Uses Verifiers for correctness evaluation
# Produces histograms and scatter plots of uncertainty vs correctness
# ============================================================

import os
import time
import gc
import random
from typing import List, Dict, Any, Tuple
import re
from getpass import getpass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from huggingface_hub import login, whoami

# Verifiers imports
import verifiers as vf
from datasets import load_dataset

print("torch:", torch.__version__)

# ============================================================
# Hugging Face Authentication
# ============================================================

print("\n" + "="*60)
print("HUGGING FACE AUTHENTICATION")
print("="*60)
print("You need a Hugging Face token to access gated models like LLaMA.")
print("Get your token from: https://huggingface.co/settings/tokens")
print("Make sure you've accepted the LLaMA license at:")
print("https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
print("="*60)

HF_TOKEN = getpass("Paste your Hugging Face token: ").strip()

try:
    login(HF_TOKEN, add_to_git_credential=False)
    user_info = whoami()
    print(f"\n✓ Successfully logged in as: {user_info.get('name', 'unknown')}")
except Exception as e:
    print(f"\n✗ Authentication failed: {e}")
    print("Please check your token and try again.")
    exit(1)

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"\nUsing device={DEVICE}, dtype={DTYPE}")

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = HF_TOKEN

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Experiment settings
MAX_TASKS = 50              # Number of HumanEval tasks to evaluate
N_PROMPTS_MTE = 5          # Number of prompt variants for MTE ensemble
N_VAL_TASKS = 10           # Tasks for training logistic regression
MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.92

# Models to evaluate
MODELS = [
    ("llama", "3B-instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("qwen-coder", "3B-instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
]

print("\n" + "="*60)
print("MODELS TO EVALUATE")
print("="*60)
for family, size, model_id in MODELS:
    print(f"  • {model_id}")
print("="*60)

# Prompt templates for BayesPE
BAYESPE_TEMPLATES = [
    "You are an expert Python developer. Write only the function implementation.\n\n{p}\n# Your solution:\n",
    "Complete the Python function so it passes the tests. No explanations.\n\n{p}\n# Implementation:\n",
    "Return a minimal and correct Python solution for this function.\n\n{p}\n# Code:\n",
    "Implement the function described below in Python. Output only valid code.\n\n{p}\n# Solution:\n",
    "Write a concise Python implementation that satisfies the specification.\n\n{p}\n# Function:\n",
]

# ============================================================
# Model Loading
# ============================================================

def load_model(model_id: str):
    """Load tokenizer and model with HF authentication"""
    print(f"\nLoading model: {model_id}")
    
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        token=HF_TOKEN  # Use the authenticated token
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        token=HF_TOKEN  # Use the authenticated token
    )
    model.eval()
    return tok, model

# ============================================================
# Prompt Construction
# ============================================================

def build_prompt_for_model(raw_prompt: str, family: str) -> Dict[str, Any]:
    """Build appropriate prompt format for each model family"""
    if family in {"llama"}:
        system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
        user = raw_prompt + "\n\n# Your code below:\n"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return {"chat": True, "messages": messages}
    
    if family in {"qwen-coder", "deepseek-coder"}:
        text = raw_prompt + "\n# Your code below:\n"
        return {"chat": False, "text": text}
    
    return {"chat": False, "text": raw_prompt + "\n# Your code below:\n"}

def apply_mte_prompt(raw_prompt: str, idx: int) -> str:
    """Apply one of the BayesPE prompt templates"""
    tpl = BAYESPE_TEMPLATES[idx % len(BAYESPE_TEMPLATES)]
    return tpl.format(p=raw_prompt.strip())

# ============================================================
# Generation with Token Entropy
# ============================================================

def token_entropies_from_logits_list(scores: List[torch.Tensor]) -> Tuple[List[float], float]:
    """Compute per-token entropy and mean entropy from logits"""
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
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.92,
) -> Dict[str, Any]:
    """Generate code with per-token entropy tracking"""
    device = next(model.parameters()).device
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
        do_sample=True,
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
    }

# ============================================================
# Verifiers Setup
# ============================================================

class HumanEvalCodeParser(vf.Parser):
    """Parser to extract code from model output"""
    def parse_answer(self, response: str) -> str:
        # Try to extract last ```python ... ``` block if present
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if code_blocks:
            code = code_blocks[-1].strip()
        else:
            # Fallback: treat whole response as code
            code = response.strip()
        
        # Basic syntax check
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError:
            return ""
        
        return code

def passes_humaneval_tests(prompt, completion, info, parser, **state):
    """Reward function: 1.0 if all tests pass, else 0.0"""
    code = parser.parse_answer(completion[-1]["content"] if isinstance(completion, list) else completion)
    if not code:
        return 0.0
    
    test_src = info["test"]
    entry_point = info["entry_point"]
    
    # Build module: prompt + code + tests
    module_src = prompt + "\n" + code + "\n\n" + test_src
    
    glb = {}
    try:
        exec(module_src, glb, glb)
        candidate_fn = glb[entry_point]
        glb["candidate"] = candidate_fn
        glb["check"](candidate_fn)
        return 1.0
    except Exception:
        return 0.0

def create_humaneval_dataset():
    """Create Verifiers-compatible HumanEval dataset"""
    raw = load_dataset("openai_humaneval", split="test")
    
    def row_to_env_record(row):
        return {
            "prompt": row["prompt"],
            "answer": row["canonical_solution"],
            "info": {
                "test": row["test"],
                "entry_point": row["entry_point"],
                "task_id": row["task_id"],
            },
        }
    
    return raw.map(row_to_env_record, remove_columns=raw.column_names)

# ============================================================
# Main Evaluation Loop
# ============================================================

def evaluate_model(family: str, size: str, model_id: str):
    """Evaluate a single model and return results"""
    print(f"\n{'='*60}")
    print(f"Model: {model_id} [{family} / {size}]")
    print(f"{'='*60}")
    
    # Load model
    try:
        t0 = time.time()
        tok, model = load_model(model_id)
        print(f"Loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None
    
    # Load HumanEval
    heval = load_dataset("openai_humaneval")["test"]
    subset = [heval[i] for i in range(min(MAX_TASKS, len(heval)))]
    print(f"Using {len(subset)} HumanEval tasks")
    
    # Storage for results
    per_task_mean_entropies: List[float] = []  # H(x_j)
    per_task_var_entropies: List[float] = []   # V(x_j)
    per_task_correctness: List[int] = []       # 1 = correct, 0 = incorrect
    per_task_codes: List[str] = []             # Generated code for examples
    per_task_prompts: List[str] = []           # Original prompts for examples
    
    # Verifiers parser
    parser = HumanEvalCodeParser()
    
    for idx, item in enumerate(subset):
        task_id = item["task_id"]
        raw_prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        
        per_prompt_codes = []
        per_prompt_mean_entropies = []
        
        print(f"  Task {idx+1}/{len(subset)} - {task_id}", flush=True)
        
        # Generate once per prompt template
        for p_idx in range(N_PROMPTS_MTE):
            wrapped_prompt = apply_mte_prompt(raw_prompt, p_idx)
            try:
                res = generate_with_entropy(
                    tok,
                    model,
                    wrapped_prompt,
                    family=family,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM on {model_id} / {task_id}")
                    res = {
                        "generated_code": "",
                        "mean_token_entropy_nats": 0.0,
                        "first5_entropies": [],
                        "num_steps": 0,
                    }
                else:
                    raise
            
            per_prompt_codes.append(res["generated_code"])
            per_prompt_mean_entropies.append(res["mean_token_entropy_nats"])
        
        # Test the first generated code for correctness using Verifiers
        first_code = per_prompt_codes[0]
        correctness = passes_humaneval_tests(
            raw_prompt, first_code, 
            {"test": test_code, "entry_point": entry_point, "task_id": task_id},
            parser
        )
        
        # Compute H(x) and V(x) for this task
        H_j = float(np.mean(per_prompt_mean_entropies))
        V_j = float(np.var(per_prompt_mean_entropies))
        
        per_task_mean_entropies.append(H_j)
        per_task_var_entropies.append(V_j)
        per_task_correctness.append(int(correctness))
        per_task_codes.append(first_code)
        per_task_prompts.append(raw_prompt)
    
    # Convert to arrays
    H_all = np.array(per_task_mean_entropies, dtype=float)
    V_all = np.array(per_task_var_entropies, dtype=float)
    y_all = np.array(per_task_correctness, dtype=int)
    n_tasks = len(y_all)
    
    # Train logistic regression on validation subset
    n_val = min(N_VAL_TASKS, n_tasks)
    perm = np.arange(n_tasks)
    np.random.shuffle(perm)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:]
    
    H_val, V_val, y_val = H_all[val_idx], V_all[val_idx], y_all[val_idx]
    
    # Standardize H & V
    H_mean, H_std = float(H_val.mean()), float(H_val.std() if H_val.std() > 1e-8 else 1.0)
    V_mean, V_std = float(V_val.mean()), float(V_val.std() if V_val.std() > 1e-8 else 1.0)
    
    H_val_std = (H_val - H_mean) / H_std
    V_val_std = (V_val - V_mean) / V_std
    X_val = np.stack([H_val_std, V_val_std], axis=1)
    
    # Fit logistic regression y ~ H,V (predict failure, so flip labels)
    y_val_fail = 1 - y_val  # 1 = fail, 0 = success
    
    if len(np.unique(y_val_fail)) > 1:
        lr = LogisticRegression()
        lr.fit(X_val, y_val_fail)
        w1, w2 = lr.coef_[0]
        b = lr.intercept_[0]
    else:
        # Degenerate case
        p = float(y_val_fail.mean()) if y_val_fail.size > 0 else 0.5
        p = min(max(p, 1e-4), 1 - 1e-4)
        w1, w2 = 0.0, 0.0
        b = np.log(p / (1 - p))
    
    print(f"\nLogistic weights: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")
    
    # Compute uncertainty for all tasks
    H_all_std = (H_all - H_mean) / H_std
    V_all_std = (V_all - V_mean) / V_std
    logits = w1 * H_all_std + w2 * V_all_std + b
    uncertainty = 1.0 / (1.0 + np.exp(-logits))  # Pfail
    
    accuracy = float(y_all.mean())
    mean_uncertainty = float(uncertainty.mean())
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Mean Uncertainty: {mean_uncertainty:.3f}")
    
    # Clean up
    del model, tok
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "family": family,
        "size": size,
        "model_id": model_id,
        "correctness": y_all,
        "uncertainty": uncertainty,
        "H": H_all,
        "V": V_all,
        "codes": per_task_codes,
        "prompts": per_task_prompts,
        "accuracy": accuracy,
        "mean_uncertainty": mean_uncertainty,
        "w1": w1,
        "w2": w2,
        "b": b,
    }

# ============================================================
# Visualization
# ============================================================

def plot_results(results: Dict[str, Any], save_prefix: str):
    """Generate histograms and scatter plots"""
    correctness = results["correctness"]
    uncertainty = results["uncertainty"]
    model_name = results["model_id"].split("/")[-1]
    
    sns.set(style="whitegrid", font_scale=1.2)
    
    # 1. Histogram of Uncertainty
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct_mask = correctness == 1
    incorrect_mask = correctness == 0
    
    ax.hist(uncertainty[correct_mask], bins=20, alpha=0.7, color='blue', 
            label=f'Correct (n={correct_mask.sum()})', edgecolor='black')
    ax.hist(uncertainty[incorrect_mask], bins=20, alpha=0.7, color='red', 
            label=f'Incorrect (n={incorrect_mask.sum()})', edgecolor='black')
    
    ax.set_xlabel('Uncertainty (Pfail)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Uncertainty Distribution - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add jitter to x-axis only for visibility (not y-axis)
    jitter_x = np.random.normal(0, 0.01, size=len(correctness))
    x_jittered = uncertainty + jitter_x
    
    colors = ['blue' if c == 1 else 'red' for c in correctness]
    ax.scatter(x_jittered, correctness, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Uncertainty (Pfail)')
    ax.set_ylabel('Correctness (0=Incorrect, 1=Correct)')
    ax.set_title(f'Uncertainty vs Correctness - {model_name}')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Correct'),
        Patch(facecolor='red', label='Incorrect')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_examples(results: Dict[str, Any]):
    """Find good and bad examples of uncertainty estimation"""
    correctness = results["correctness"]
    uncertainty = results["uncertainty"]
    codes = results["codes"]
    prompts = results["prompts"]
    
    # Good example 1: Correct + Low Uncertainty
    correct_low_unc_idx = np.where((correctness == 1) & (uncertainty < 0.3))[0]
    if len(correct_low_unc_idx) > 0:
        idx = correct_low_unc_idx[np.argmin(uncertainty[correct_low_unc_idx])]
        print("\n" + "="*60)
        print("GOOD EXAMPLE 1: Correct Output + Low Uncertainty")
        print("="*60)
        print(f"Uncertainty: {uncertainty[idx]:.4f}")
        print(f"Correctness: {correctness[idx]}")
        print(f"\nPrompt:\n{prompts[idx][:200]}...")
        print(f"\nGenerated Code:\n{codes[idx][:300]}...")
        print("\nWhy this makes sense: The model generated correct code and had low")
        print("uncertainty across prompt variants, indicating confidence in the solution.")
    
    # Good example 2: Incorrect + High Uncertainty
    incorrect_high_unc_idx = np.where((correctness == 0) & (uncertainty > 0.7))[0]
    if len(incorrect_high_unc_idx) > 0:
        idx = incorrect_high_unc_idx[np.argmax(uncertainty[incorrect_high_unc_idx])]
        print("\n" + "="*60)
        print("GOOD EXAMPLE 2: Incorrect Output + High Uncertainty")
        print("="*60)
        print(f"Uncertainty: {uncertainty[idx]:.4f}")
        print(f"Correctness: {correctness[idx]}")
        print(f"\nPrompt:\n{prompts[idx][:200]}...")
        print(f"\nGenerated Code:\n{codes[idx][:300]}...")
        print("\nWhy this makes sense: The model struggled with this problem and showed")
        print("high uncertainty across prompt variants, correctly signaling its uncertainty.")
    
    # Failure case: Incorrect + Low Uncertainty (Overconfident)
    incorrect_low_unc_idx = np.where((correctness == 0) & (uncertainty < 0.3))[0]
    if len(incorrect_low_unc_idx) > 0:
        idx = incorrect_low_unc_idx[np.argmin(uncertainty[incorrect_low_unc_idx])]
        print("\n" + "="*60)
        print("FAILURE CASE: Incorrect Output + Low Uncertainty (Overconfident)")
        print("="*60)
        print(f"Uncertainty: {uncertainty[idx]:.4f}")
        print(f"Correctness: {correctness[idx]}")
        print(f"\nPrompt:\n{prompts[idx][:200]}...")
        print(f"\nGenerated Code:\n{codes[idx][:300]}...")
        print("\nWhy this failed: The model was overconfident - it generated similar")
        print("incorrect code across all prompt variants, showing low variance in token")
        print("entropies. This suggests the model has a systematic misunderstanding of")
        print("the problem rather than genuine uncertainty.")

# ============================================================
# Run Evaluation
# ============================================================

if __name__ == "__main__":
    all_results = []
    
    for family, size, model_id in MODELS:
        results = evaluate_model(family, size, model_id)
        if results is not None:
            all_results.append(results)
            
            # Generate plots
            save_prefix = f"{family}_{size}".replace(".", "p")
            plot_results(results, save_prefix)
            
            # Print examples
            find_examples(results)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL MODELS")
    print("="*60)
    for r in all_results:
        print(f"\n{r['model_id']}")
        print(f"  Accuracy: {r['accuracy']:.3f}")
        print(f"  Mean Uncertainty: {r['mean_uncertainty']:.3f}")
        print(f"  Weights: w1={r['w1']:.4f}, w2={r['w2']:.4f}, b={r['b']:.4f}")
    
    # Save results
    summary_df = pd.DataFrame([{
        "model": r["model_id"],
        "family": r["family"],
        "accuracy": r["accuracy"],
        "mean_uncertainty": r["mean_uncertainty"],
        "w1": r["w1"],
        "w2": r["w2"],
        "b": r["b"],
    } for r in all_results])
    
    summary_df.to_csv("mte_uncertainty_results.csv", index=False)
    print("\nSaved: mte_uncertainty_results.csv")