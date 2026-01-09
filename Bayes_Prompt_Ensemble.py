%matplotlib inline
!pip install -q \
  "transformers==4.46.2" \
  "accelerate==0.34.2" \
  "sentencepiece" \
  "datasets==2.20.0" \
  "pandas==2.2.2" \
  "pyarrow<20" \
  "human-eval" \
  "verifiers" \
  "tqdm" \
  "scikit-learn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json
import gc
import time
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

print("torch:", torch.__version__)

# -------------------------
# Authentication
# -------------------------
from huggingface_hub import login, whoami
from getpass import getpass

HF_TOKEN = getpass("Paste your Hugging Face token: ").strip()
login(HF_TOKEN, add_to_git_credential=False)
print("Logged in as:", whoami().get("name", "unknown"))

# -------------------------
# Load Dataset
# -------------------------
heval = load_dataset("openai_humaneval")["test"]
# Increased from 50 to 164 for meaningful evaluation
subset = [heval[i] for i in range(len(heval))]
print(f"Using {len(subset)} HumanEval tasks (full dataset).")
print("Example task_id:", subset[0]["task_id"])

# -------------------------
# Model Definitions
# -------------------------
MODELS = [
    ("llama", "3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("qwen-coder-instruct", "3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
]

# -------------------------
# Multiple Prompts for Ensemble
# -------------------------
PROMPTS = [
    # Prompt 1: Standard instruction
    "You are a Python coding assistant. Complete the function so that it passes the tests. Return only Python code, no explanation.",
    
    # Prompt 2: Step-by-step reasoning
    "You are an expert Python programmer. Carefully analyze the function requirements and implement a correct solution. Think step by step, then provide only the Python code.",
    
    # Prompt 3: Focus on edge cases
    "You are a meticulous Python developer. Consider all edge cases and implement a robust solution. Output only valid Python code for the function.",
    
    # Prompt 4: Concise instruction
    "Complete this Python function correctly. Return only code, no comments or explanation.",
    
    # Prompt 5: Quality emphasis
    "You are a senior software engineer. Write clean, correct Python code for this function. Provide only the implementation.",
]

print(f"Using {len(PROMPTS)} different prompts for ensemble")

# -------------------------
# Model Loading Function
# -------------------------
def load_model(model_id: str, hf_token: str = None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return tok, model

# -------------------------
# Prompt Building Function
# -------------------------
def build_prompt_for_model(raw_prompt: str, system_prompt: str, family: str):
    """Build appropriate prompt format for each model family."""
    if family in {"llama", "deepseek", "qwen-coder-instruct"}:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_prompt + "\n\n# Your code below:\n"},
        ]
        return {"chat": True, "messages": messages}
    
    return {"chat": False, "text": raw_prompt + "\n# Your code below:\n"}

# -------------------------
# Token Entropy Calculation
# -------------------------
def token_entropies_from_logits_list(scores):
    """Calculate per-token entropies from logits."""
    H = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        H.append(ent)
    return H, (float(np.mean(H)) if H else float("nan"))

# -------------------------
# Generation with Entropy Tracking
# -------------------------
@torch.inference_mode()
def generate_with_entropy(tok, model, raw_prompt: str, system_prompt: str, family: str,
                          max_new_tokens: int = 320, temperature: float = 0.0, top_p: float = 1.0):
    """Generate code and track token entropies."""
    device = model.device
    spec = build_prompt_for_model(raw_prompt, system_prompt, family)
    
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
        do_sample=False if temperature == 0.0 else True,
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
        "entropies": H_list,
        "num_steps": len(H_list),
    }

# -------------------------
# Verifiers Setup
# -------------------------
import verifiers as vf
import sys
import io
import signal
from contextlib import redirect_stdout, redirect_stderr
import os

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

verifiers_dataset = raw.map(row_to_env_record, remove_columns=raw.column_names)

class HumanEvalCodeParser(vf.Parser):
    def parse_answer(self, response: str) -> str:
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if code_blocks:
            code = code_blocks[-1].strip()
        else:
            code = response.strip()
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError:
            return ""
        return code

def _run_test_with_timeout(module_src, entry_point, timeout_seconds=10):
    f = io.StringIO()
    use_timeout = hasattr(signal, 'SIGALRM') and os.name != 'nt'
    old_handler = None
    
    if use_timeout:
        try:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test execution exceeded {timeout_seconds} seconds")
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        except (ValueError, OSError, AttributeError):
            use_timeout = False
    
    try:
        with redirect_stdout(f), redirect_stderr(f):
            glb = {}
            exec(module_src, glb, glb)
            candidate_fn = glb[entry_point]
            glb["candidate"] = candidate_fn
            glb["check"](candidate_fn)
        return True, None
    except TimeoutError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)
    finally:
        if use_timeout and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def passes_humaneval_tests(prompt, completion, info, parser, **state):
    code = parser.parse_answer(completion[-1]["content"] if isinstance(completion, list) else completion)
    if not code:
        return 0.0
    
    test_src = info["test"]
    entry_point = info["entry_point"]
    module_src = prompt + "\n" + code + "\n\n" + test_src
    
    success, error = _run_test_with_timeout(module_src, entry_point, timeout_seconds=10)
    return 1.0 if success else 0.0

def evaluate_completions_with_verifiers(completions_file: str):
    parser = HumanEvalCodeParser()
    completions = {}
    with open(completions_file, "r") as f:
        for line in f:
            item = json.loads(line)
            completions[item["task_id"]] = item["completion"]
    
    results = {}
    for task_id, completion_code in tqdm(completions.items(), desc="Evaluating", unit="task"):
        dataset_entry = None
        for entry in verifiers_dataset:
            if entry["info"]["task_id"] == task_id:
                dataset_entry = entry
                break
        
        if dataset_entry is None:
            results[task_id] = False
            continue
        
        reward = passes_humaneval_tests(
            prompt=dataset_entry["prompt"],
            completion=completion_code,
            info=dataset_entry["info"],
            parser=parser
        )
        results[task_id] = (reward == 1.0)
    
    return results

print("✅ Verifiers setup complete!")

# -------------------------
# Checkpoint Management
# -------------------------
import pickle
from pathlib import Path

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

def save_checkpoint(data, filename):
    """Save checkpoint data."""
    filepath = CHECKPOINT_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Saved checkpoint: {filepath}")

def load_checkpoint(filename):
    """Load checkpoint data if exists."""
    filepath = CHECKPOINT_DIR / filename
    if filepath.exists():
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"📂 Loaded checkpoint: {filepath}")
        return data
    return None

# Try to load existing results
existing_rows = load_checkpoint("generation_results.pkl")
if existing_rows is not None:
    print(f"✅ Found existing results with {len(existing_rows)} rows")
    rows = existing_rows
    SKIP_GENERATION = True
else:
    print("🔄 No existing results found. Starting fresh generation...")
    rows = []
    SKIP_GENERATION = False

# -------------------------
# Main Generation Loop with Multiple Prompts
# -------------------------
if not SKIP_GENERATION:

  for family, size_bucket, model_id in MODELS:
    print(f"\n{'='*80}")
    print(f"Loading model: {model_id}  [{family} / {size_bucket}]")
    print(f"{'='*80}")
    
    try:
        t0 = time.time()
        tok, model = load_model(model_id, hf_token=HF_TOKEN)
        print(f"Loaded in {time.time() - t0:.1f}s")
    except RuntimeError as e:
        print(f"Failed to load {model_id}: {e}")
        continue
    
    for item in tqdm(subset, desc=f"Generating for {model_id.split('/')[-1]}", unit="task"):
        task_id = item["task_id"]
        prompt = item["prompt"]
        
        # Generate with EACH prompt and track entropies
        prompt_entropies = []
        prompt_codes = []
        
        for prompt_idx, system_prompt in enumerate(PROMPTS):
            try:
                res = generate_with_entropy(
                    tok, model, prompt, system_prompt=system_prompt, family=family,
                    max_new_tokens=320, temperature=0.0, top_p=1.0
                )
                
                # Clean generated code
                generated_code = res["generated_code"]
                generated_code = re.sub(r'^```python\s*\n', '', generated_code, flags=re.MULTILINE)
                generated_code = re.sub(r'^```\s*\n', '', generated_code, flags=re.MULTILINE)
                generated_code = re.sub(r'\n```\s*$', '', generated_code, flags=re.MULTILINE)
                generated_code = generated_code.strip()
                
                prompt_entropies.append(res["mean_token_entropy_nats"])
                prompt_codes.append(generated_code)
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM for {model_id} / {task_id}. Skipping.")
                    break
                else:
                    raise
        
        if len(prompt_entropies) < len(PROMPTS):
            break  # OOM occurred
        
        # Compute H(x) and V(x) for ensemble
        H_x = np.mean(prompt_entropies)  # Mean of mean token entropies
        V_x = np.var(prompt_entropies)   # Variance of mean token entropies
        
        # Use the FIRST prompt's generation for evaluation
        solution_code = prompt_codes[0]
        
        rows.append({
            "family": family,
            "size_bucket": size_bucket,
            "model_id": model_id,
            "task_id": task_id,
            "H_x": H_x,  # Mean of entropies
            "V_x": V_x,  # Variance of entropies
            "prompt_entropies": prompt_entropies,
            "solution_code": solution_code,
            "preview": (solution_code[:180].replace("\n", " \\n ") +
                       ("..." if len(solution_code) > 180 else "")),
        })
    
    # Evaluate solutions using verifiers
    if len(rows) > 0:
        model_completions_file = f"humaneval_completions_{model_id.replace('/', '_')}_ensemble.jsonl"
        model_rows = [r for r in rows if r["model_id"] == model_id]
        
        with open(model_completions_file, "w") as f:
            for row in model_rows:
                f.write(json.dumps({
                    "task_id": row["task_id"],
                    "completion": row["solution_code"]
                }) + "\n")
        
        print(f"\nRunning verifier for {model_id}...")
        try:
            verifier_results = evaluate_completions_with_verifiers(model_completions_file)
            
            # Add correctness to rows
            for row in rows:
                if row["model_id"] == model_id and row["task_id"] in verifier_results:
                    row["correctness"] = 1 if verifier_results[row["task_id"]] else 0
            
            pass_rate = sum(verifier_results.values()) / len(verifier_results) * 100
            print(f"✅ Pass rate: {sum(verifier_results.values())}/{len(verifier_results)} ({pass_rate:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Free memory
    del model
    del tok
    torch.cuda.empty_cache()
    gc.collect()

df = pd.DataFrame(rows)
print(f"\n✅ Collected {len(df)} rows.")

# -------------------------
# Train Logistic Regression Model
# -------------------------
print("\n" + "="*80)
print("TRAINING UNCERTAINTY MODEL (Logistic Regression)")
print("="*80)

# Prepare data for training
train_data = df[df["correctness"].notna()].copy()
print(f"Training data: {len(train_data)} examples")

# Create binary labels: y=1 if fail, y=0 if success
train_data["y"] = train_data["correctness"].apply(lambda x: 0 if x == 1 else 1)

# Features: H(x) and V(x)
X = train_data[["H_x", "V_x"]].values
y = train_data["y"].values

print(f"Features shape: {X.shape}")
print(f"Labels distribution: {np.bincount(y.astype(int))}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_scaled, y)

# Get parameters
w1, w2 = lr_model.coef_[0]
b = lr_model.intercept_[0]

print(f"\nTrained parameters:")
print(f"  w1 (weight for H): {w1:.4f}")
print(f"  w2 (weight for V): {w2:.4f}")
print(f"  b (bias): {b:.4f}")

# Evaluate model accuracy
y_pred = lr_model.predict(X_scaled)
y_pred_proba = lr_model.predict_proba(X_scaled)[:, 1]

accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_pred_proba)

print(f"\n📊 Model Performance:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  AUC-ROC: {auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Success", "Failure"]))

# Add uncertainty scores to dataframe
df["U_x"] = np.nan
for idx, row in df.iterrows():
    if not pd.isna(row["H_x"]) and not pd.isna(row["V_x"]):
        x_vec = scaler.transform([[row["H_x"], row["V_x"]]])
        P_fail = lr_model.predict_proba(x_vec)[0, 1]
        df.at[idx, "U_x"] = P_fail

print(f"\n✅ Uncertainty scores computed for all {len(df)} examples")

# Save results
df.to_csv("humaneval_prompt_ensemble_results.csv", index=False)
print(f"✅ Saved results to humaneval_prompt_ensemble_results.csv")

# -------------------------
# Visualization
# -------------------------
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

def create_uncertainty_plots(df: pd.DataFrame, output_dir: str = "."):
    models = df["model_id"].unique()
    
    for model_id in models:
        model_data = df[(df["model_id"] == model_id) & (df["correctness"].notna())].copy()
        
        if len(model_data) == 0:
            print(f"⚠️  Skipping {model_id}: No data")
            continue
        
        model_name_short = model_id.split("/")[-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        correct_unc = model_data[model_data["correctness"] == 1]["U_x"]
        incorrect_unc = model_data[model_data["correctness"] == 0]["U_x"]
        
        all_unc = model_data["U_x"]
        bins = np.linspace(all_unc.min(), all_unc.max(), 30)
        
        if len(correct_unc) > 0:
            ax1.hist(correct_unc, bins=bins, alpha=0.6, color='blue',
                    label=f'Correct (n={len(correct_unc)})', edgecolor='black', linewidth=0.5)
        if len(incorrect_unc) > 0:
            ax1.hist(incorrect_unc, bins=bins, alpha=0.6, color='red',
                    label=f'Incorrect (n={len(incorrect_unc)})', edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Uncertainty Score U(x) = P(fail)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Histogram of Uncertainty: {model_name_short}', fontsize=12, weight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Scatter plot
        correct_mask = model_data["correctness"] == 1
        incorrect_mask = model_data["correctness"] == 0
        
        if correct_mask.sum() > 0:
            ax2.scatter(model_data[correct_mask]["U_x"],
                       model_data[correct_mask]["correctness"],
                       alpha=0.6, color='blue', s=50, label='Correct',
                       edgecolors='black', linewidths=0.5)
        if incorrect_mask.sum() > 0:
            ax2.scatter(model_data[incorrect_mask]["U_x"],
                       model_data[incorrect_mask]["correctness"],
                       alpha=0.6, color='red', s=50, label='Incorrect',
                       edgecolors='black', linewidths=0.5)
        
        ax2.set_xlabel('Uncertainty Score U(x) = P(fail)', fontsize=11)
        ax2.set_ylabel('Correctness (0 = Incorrect, 1 = Correct)', fontsize=11)
        ax2.set_title(f'Uncertainty vs Correctness: {model_name_short}', fontsize=12, weight='bold')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Incorrect', 'Correct'])
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{output_dir}/{model_name_short.replace('/', '_')}_uncertainty_plots_ensemble.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plots for {model_name_short}")
        plt.show()

create_uncertainty_plots(df)

# -------------------------
# Find Examples
# -------------------------
print("\n" + "="*80)
print("FINDING EXAMPLES FOR EACH MODEL")
print("="*80)

def find_examples(df: pd.DataFrame, model_id: str):
    model_data = df[(df["model_id"] == model_id) & (df["correctness"].notna())].copy()
    
    if len(model_data) == 0:
        print(f"\n⚠️  No data for {model_id}")
        return
    
    low_threshold = model_data["U_x"].quantile(0.25)
    high_threshold = model_data["U_x"].quantile(0.75)
    
    print(f"\n{'='*80}")
    print(f"Examples for {model_id}")
    print(f"{'='*80}")
    print(f"Low uncertainty threshold: {low_threshold:.4f} (25th percentile)")
    print(f"High uncertainty threshold: {high_threshold:.4f} (75th percentile)")
    
    # Correct case
    correct_low = model_data[(model_data["correctness"] == 1) &
                             (model_data["U_x"] <= low_threshold)]
    incorrect_high = model_data[(model_data["correctness"] == 0) &
                                (model_data["U_x"] >= high_threshold)]
    
    if len(correct_low) > 0:
        ex = correct_low.iloc[0]
        print(f"\n✅ CORRECT CASE (Correct + Low Uncertainty):")
        print(f"   Task: {ex['task_id']}")
        print(f"   U(x): {ex['U_x']:.4f}, H(x): {ex['H_x']:.4f}, V(x): {ex['V_x']:.6f}")
        print(f"   Code preview: {ex['solution_code'][:200]}...")
        print(f"   Explanation: Low uncertainty correctly reflects high confidence in correct solution.")
    elif len(incorrect_high) > 0:
        ex = incorrect_high.iloc[0]
        print(f"\n✅ CORRECT CASE (Incorrect + High Uncertainty):")
        print(f"   Task: {ex['task_id']}")
        print(f"   U(x): {ex['U_x']:.4f}, H(x): {ex['H_x']:.4f}, V(x): {ex['V_x']:.6f}")
        print(f"   Code preview: {ex['solution_code'][:200]}...")
        print(f"   Explanation: High uncertainty correctly reflects low confidence.")
    
    # Failure case
    incorrect_low = model_data[(model_data["correctness"] == 0) &
                               (model_data["U_x"] <= low_threshold)]
    
    if len(incorrect_low) > 0:
        ex = incorrect_low.iloc[0]
        print(f"\n❌ FAILURE CASE (Incorrect + Low Uncertainty - Overconfident):")
        print(f"   Task: {ex['task_id']}")
        print(f"   U(x): {ex['U_x']:.4f}, H(x): {ex['H_x']:.4f}, V(x): {ex['V_x']:.6f}")
        print(f"   Code preview: {ex['solution_code'][:200]}...")
        print(f"   Explanation: Model is overconfident despite incorrect solution.")
        print(f"   Possible causes: Similar training patterns, syntactically correct but")
        print(f"   logically wrong, or consistent errors across all prompts.")

for model_id in df["model_id"].unique():
    find_examples(df, model_id)

print("\n" + "="*80)
print("✅ COMPLETE!")
print("="*80)