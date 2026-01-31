#!/usr/bin/env python3
"""
Train TBG+MLP and SLT+MLP Probes

This script:
1. Samples code ONCE (M_samples=20) for all methods
2. Extracts features using SLT and TBG methods
3. Trains MLP classifiers for both
4. Saves probes and dataset splits
"""

import os
import re
import io
import gc
import json
import time
import random
import hashlib
import signal
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass

# ============================================================
# Configuration
# ============================================================

def get_experiment_config():
    return {
        # Models to train probes for (from MTEmodels notebook)
        "models": [
            ("llama", "3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
            ("qwen-coder-instruct", "3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
            ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
        ],
        "dataset_name": "openai_humaneval",
        "split": "test",
        "limit_tasks": None,  # None = all 164 tasks
        
        # Sampling (done ONCE for all methods)
        "M_samples": 20,
        
        "sample_max_new_tokens": 256,
        "sample_temperature": 0.7,
        "sample_top_p": 0.95,
        "greedy_max_new_tokens": 256,
        "test_timeout_s": 10,
        "seed": 42,
        
        # Feature extraction methods to use (only SLT and TBG)
        "feature_methods": ["SLT", "TBG"],
        
        # Layers to use
        "layers": [-3, -2, -1],
        
        # Classifier (only MLP)
        "classifier": "mlp",
        
        # Labeling
        "label_mode": "median",  # Use all examples - median split for binary classification
    }

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function so that it passes the tests. "
    "Return only Python code, no explanation."
)

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Helper Functions
# ============================================================

def build_prompt_for_model(raw_prompt: str, family: str):
    """Build prompt based on model family (from MTEmodels notebook)."""
    if family in {"llama", "deepseek"}:
        system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
        user = raw_prompt + "\n\n# Your code below:\n"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return {"chat": True, "messages": messages}
    
    if family == "qwen-coder" or family == "qwen-coder-instruct":
        system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
        user = raw_prompt + "\n\n# Your code below:\n"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return {"chat": True, "messages": messages}
    
    return {"chat": False, "text": raw_prompt + "\n# Your code below:\n"}

def build_chat_text(tok, user_prompt: str, family: str = "llama"):
    """Build chat text for model, handling different model families."""
    spec = build_prompt_for_model(user_prompt, family)
    has_chat_template = getattr(tok, "chat_template", None) not in (None, "")
    
    if spec["chat"] and has_chat_template:
        return tok.apply_chat_template(
            spec["messages"], add_generation_prompt=True, tokenize=False
        )
    elif spec["chat"]:
        # Fallback: build text manually
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
        return "".join(parts) + "\n[ASSISTANT]\n"
    else:
        return spec["text"]

def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = blocks[-1].strip() if blocks else text.strip()
    code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```\s*$", "", code)
    try:
        compile(code, "<candidate>", "exec")
        return code
    except SyntaxError:
        return ""

def _run_test_with_timeout(module_src: str, entry_point: str, timeout_seconds: int = 10):
    f = io.StringIO()
    use_timeout = hasattr(signal, "SIGALRM") and os.name != "nt"
    old_handler = None
    if use_timeout:
        try:
            def handler(signum, frame):
                raise TimeoutError(f"timeout>{timeout_seconds}s")
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
        except Exception:
            use_timeout = False
    try:
        with redirect_stdout(f), redirect_stderr(f):
            glb = {}
            exec(module_src, glb, glb)
            fn = glb[entry_point]
            glb["check"](fn)
        return True, f.getvalue()
    except Exception as e:
        return False, f.getvalue() + "\n" + repr(e)
    finally:
        if use_timeout and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def semantic_signature(prompt_src: str, test_src: str, entry_point: str, code: str, timeout_s: int = 10):
    if not code:
        return "INVALID:syntax", 0
    module_src = prompt_src + "\n" + code + "\n\n" + test_src
    ok, logs = _run_test_with_timeout(module_src, entry_point, timeout_seconds=timeout_s)
    logs_norm = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", logs)
    h = hashlib.sha256(logs_norm.encode("utf-8", errors="ignore")).hexdigest()[:16]
    sig = ("PASS" if ok else "FAIL") + ":" + h
    return sig, int(ok)

def softmax_probs_from_len_norm_logps(logps):
    a = np.array(logps, dtype=np.float64)
    a = a - np.max(a)
    p = np.exp(a)
    denom = np.sum(p)
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(p) / max(len(p), 1)
    return p / denom

def entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))

# ============================================================
# Model Loading
# ============================================================

def load_model(model_id: str, hf_token: str | None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
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
    except GatedRepoError as e:
        print(f"[SKIP] Gated repo (no access): {model_id}\n  {e}")
        return None, None
    except OSError as e:
        msg = str(e)
        if "gated repo" in msg.lower() or "401" in msg or "403" in msg or "unauthorized" in msg.lower():
            print(f"[SKIP] Unauthorized / gated model: {model_id}\n  {e}")
            return None, None
        raise

# ============================================================
# Generation Functions
# ============================================================

@torch.inference_mode()
def sample_completion_with_len_norm_logp(tok, model, chat_text: str, max_new_tokens: int, temperature: float, top_p: float):
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full_ids = out.sequences[0]
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = full_ids[prompt_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    
    logps = []
    for t, step_logits in enumerate(out.scores):
        if t >= len(gen_ids):
            break
        token_id = gen_ids[t].item()
        lprobs = torch.log_softmax(step_logits[0], dim=-1)
        logps.append(lprobs[token_id].item())
    
    L = max(len(logps), 1)
    len_norm_logp = float(np.sum(logps) / L)
    return {"text": gen_text, "len_norm_logp": len_norm_logp}

@torch.inference_mode()
def greedy_generate_full_ids_and_prompt_len(tok, model, chat_text: str, max_new_tokens: int):
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full_ids = out.sequences[0].detach().cpu()
    prompt_len = int(enc["input_ids"].shape[1])
    return full_ids, prompt_len

# ============================================================
# Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_features_multi_method(tok, model, full_ids_cpu: torch.Tensor, prompt_len: int, 
                                   layers: list, method: str):
    """
    Extract features using different methods:
    - SLT: second-to-last token (index -2)
    - TBG: token before generation (prompt_len - 1)
    """
    full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
    out = model(full_ids, output_hidden_states=True, use_cache=False)
    
    features = []
    for layer in layers:
        hs = out.hidden_states[layer]
        
        if method == "SLT":
            token_idx = -2
        elif method == "TBG":
            token_idx = prompt_len - 1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle edge cases
        if token_idx < 0:
            token_idx = hs.shape[1] + token_idx
        
        if token_idx >= hs.shape[1]:
            token_idx = hs.shape[1] - 1
        
        if token_idx < 0:
            token_idx = 0
        
        features.append(hs[0, token_idx, :].float().detach().cpu().numpy())
    
    return np.concatenate(features)

# ============================================================
# Dataset Building
# ============================================================

def build_dataset_once(tok, model, tasks, cfg, family: str):
    """Sample ONCE for all methods, then extract features using different methods."""
    rows = []
    
    for ex in tqdm(tasks, desc="Building dataset", unit="task"):
        task_id = ex["task_id"]
        prompt_src = ex["prompt"]
        test_src = ex["test"]
        entry_point = ex["entry_point"]
        
        user_prompt = prompt_src
        chat_text = build_chat_text(tok, user_prompt, family=family)
        
        # 1) Sample M completions ONCE (for semantic entropy)
        samples = []
        for _ in range(cfg["M_samples"]):
            s = sample_completion_with_len_norm_logp(
                tok, model, chat_text,
                max_new_tokens=cfg["sample_max_new_tokens"],
                temperature=cfg["sample_temperature"],
                top_p=cfg["sample_top_p"],
            )
            code = extract_code(s["text"])
            sig, ok = semantic_signature(prompt_src, test_src, entry_point, code, timeout_s=cfg["test_timeout_s"])
            samples.append({"sig": sig, "ok": ok, "len_norm_logp": s["len_norm_logp"]})
        
        # Compute semantic entropy
        p_sample = softmax_probs_from_len_norm_logps([s["len_norm_logp"] for s in samples])
        cluster_mass = {}
        for s, p in zip(samples, p_sample):
            cluster_mass[s["sig"]] = cluster_mass.get(s["sig"], 0.0) + float(p)
        
        cluster_probs = np.array(list(cluster_mass.values()), dtype=np.float64)
        if cluster_probs.sum() <= 0 or not np.isfinite(cluster_probs.sum()):
            cluster_probs = np.ones_like(cluster_probs) / max(len(cluster_probs), 1)
        else:
            cluster_probs = cluster_probs / cluster_probs.sum()
        
        sem_ent = entropy_nats(cluster_probs)
        
        # 2) Greedy completion (for features and pass@1)
        full_ids, prompt_len = greedy_generate_full_ids_and_prompt_len(
            tok, model, chat_text, max_new_tokens=cfg["greedy_max_new_tokens"]
        )
        gen_ids = full_ids[prompt_len:]
        greedy_text = tok.decode(gen_ids, skip_special_tokens=True)
        greedy_code = extract_code(greedy_text)
        _, pass_at_1 = semantic_signature(prompt_src, test_src, entry_point, greedy_code, timeout_s=cfg["test_timeout_s"])
        
        # Extract features using SLT and TBG methods
        features_dict = {}
        for method in cfg["feature_methods"]:
            feat = extract_features_multi_method(
                tok, model, full_ids, prompt_len, cfg["layers"], method
            )
            features_dict[method] = feat
        
        rows.append({
            "task_id": task_id,
            "semantic_entropy": float(sem_ent),
            "pass_at_1": int(pass_at_1),
            "prompt_len": prompt_len,
            **features_dict
        })
    
    return rows

def make_labels(semantic_entropy_values, label_mode="median"):
    """Create binary labels from semantic entropy."""
    sE = np.array(semantic_entropy_values, dtype=np.float64)
    
    if label_mode == "q25q75":
        q25, q75 = np.quantile(sE, [0.25, 0.75])
        keep = (sE <= q25) | (sE >= q75)
        y = (sE[keep] >= float(q75)).astype(int)
        return y, keep, float(q75)
    else:  # median
        thr = float(np.nanmedian(sE))
        y = (sE > thr).astype(int)
        keep = np.ones_like(y, dtype=bool)
        return y, keep, thr

# ============================================================
# Save Dataset Splits
# ============================================================

def save_dataset_splits(df_use, y, indices_train, indices_val, indices_test, output_dir):
    """Save train/val/test task IDs to CSV files."""
    split_dir = output_dir / "DatasetSplit"
    split_dir.mkdir(exist_ok=True)
    
    # Extract task IDs for each split
    train_task_ids = df_use.iloc[indices_train]["task_id"].tolist()
    val_task_ids = df_use.iloc[indices_val]["task_id"].tolist()
    test_task_ids = df_use.iloc[indices_test]["task_id"].tolist()
    
    # Save to CSV
    train_df = pd.DataFrame({"task_id": train_task_ids})
    val_df = pd.DataFrame({"task_id": val_task_ids})
    test_df = pd.DataFrame({"task_id": test_task_ids})
    
    train_df.to_csv(split_dir / "train_tasks.csv", index=False)
    val_df.to_csv(split_dir / "val_tasks.csv", index=False)
    test_df.to_csv(split_dir / "test_tasks.csv", index=False)
    
    # Save summary
    summary = {
        "total_tasks": len(df_use),
        "train_count": len(train_task_ids),
        "val_count": len(val_task_ids),
        "test_count": len(test_task_ids),
        "train_ratio": len(train_task_ids) / len(df_use) if len(df_use) > 0 else 0,
        "val_ratio": len(val_task_ids) / len(df_use) if len(df_use) > 0 else 0,
        "test_ratio": len(test_task_ids) / len(df_use) if len(df_use) > 0 else 0,
    }
    
    with open(split_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Dataset splits saved to {split_dir}/")
    print(f"   - train_tasks.csv: {len(train_task_ids)} tasks")
    print(f"   - val_tasks.csv: {len(val_task_ids)} tasks")
    print(f"   - test_tasks.csv: {len(test_task_ids)} tasks")
    print(f"   - split_summary.json: Split statistics")
    
    return train_task_ids, val_task_ids, test_task_ids

# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    """Train TBG+MLP and SLT+MLP probes for all models and save dataset splits."""
    cfg = get_experiment_config()
    set_seed(cfg["seed"])
    
    start_time = time.time()
    
    # Output directory
    output_dir = Path(__file__).parent
    probes_dir = output_dir / "saved_probes"
    probes_dir.mkdir(exist_ok=True)
    
    # Load HuggingFace token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with model access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Load dataset (same for all models)
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    tasks = [ds[i] for i in range(len(ds))]
    if cfg["limit_tasks"]:
        tasks = tasks[:cfg["limit_tasks"]]
    
    n_tasks = len(tasks)
    print(f"\n{'='*80}")
    print("PROBE TRAINING: TBG+MLP and SLT+MLP for All Models")
    print(f"{'='*80}")
    print(f"Models: {len(cfg['models'])}")
    for family, size, model_id in cfg["models"]:
        print(f"  - {model_id} ({family})")
    print(f"Tasks: {n_tasks}")
    print(f"M_samples: {cfg['M_samples']}")
    print(f"Feature methods: {cfg['feature_methods']}")
    print(f"Classifier: {cfg['classifier']}")
    print(f"{'='*80}\n")
    
    # Results storage (across all models)
    all_results = []
    split_saved = False
    
    # Train probes for each model
    for model_idx, (family, size_bucket, model_id) in enumerate(cfg["models"]):
        print(f"\n{'='*80}")
        print(f"MODEL {model_idx + 1}/{len(cfg['models'])}: {model_id}")
        print(f"Family: {family}")
        print(f"{'='*80}\n")
        
        # Load model
        print(f"⏱️  Loading model {model_id}...")
        model_start = time.time()
        tok, model = load_model(model_id, HF_TOKEN)
        if tok is None:
            print(f"❌ Failed to load {model_id}, skipping...")
            continue
        model_time = time.time() - model_start
        print(f"✅ Model loaded in {model_time:.1f}s")
        
        # Build dataset for this model
        print(f"\nStep 1: Building dataset for {model_id}...")
        print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        dataset_start = time.time()
        rows = build_dataset_once(tok, model, tasks, cfg, family=family)
        dataset_time = time.time() - dataset_start
        df = pd.DataFrame(rows)
        
        print(f"\n✅ Collected {len(df)} examples in {dataset_time/60:.1f} minutes ({dataset_time:.0f}s)")
        print(f"   Semantic entropy range: {df['semantic_entropy'].min():.4f} to {df['semantic_entropy'].max():.4f}")
        
        # Save dataset for this model
        model_name_safe = model_id.replace("/", "_")
        dataset_file = output_dir / f"probe_dataset_{model_name_safe}.csv"
        df.to_csv(dataset_file, index=False)
        print(f"✅ Dataset saved to {dataset_file}")
        
        # Cleanup model
        del model, tok
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create labels
        y_full, keep_mask, thr = make_labels(df["semantic_entropy"].values, cfg["label_mode"])
        df_use = df[keep_mask].reset_index(drop=True)
        y = y_full
        
        print(f"\nLabel distribution: y0={np.sum(y==0)}, y1={np.sum(y==1)}")
        print(f"Semantic entropy threshold: {thr:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Step 2: Training probes for {model_id}...")
        print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        training_start = time.time()
        
        # Train probes for this model
        for feature_method in cfg["feature_methods"]:
            print(f"\n{'─'*80}")
            print(f"Feature Method: {feature_method}")
            print(f"{'─'*80}")
            
            # Extract features for this method
            X = np.stack(df_use[feature_method].values).astype(np.float32)
            print(f"Feature dimensions: {X.shape[1]}")
            
            # Split data
            indices = np.arange(len(X))
            idx_train, idx_tmp, y_train, y_tmp = train_test_split(
                indices, y, test_size=0.30, random_state=cfg["seed"], stratify=y
            )
            idx_val, idx_test, y_val, y_test = train_test_split(
                idx_tmp, y_tmp, test_size=0.50, random_state=cfg["seed"], stratify=y_tmp
            )
            
            # Save dataset splits (only once, using first model's first method's split)
            if not split_saved:
                save_dataset_splits(df_use, y, idx_train, idx_val, idx_test, output_dir)
                split_saved = True
            
            X_train = X[idx_train]
            X_val = X[idx_val]
            X_test = X[idx_test]
            
            print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            # Train MLP classifier
            print(f"\n  Training MLP classifier...")
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=1000, random_state=cfg["seed"],
                early_stopping=True, validation_fraction=0.1, verbose=False
            )
            clf.fit(X_train_s, y_train)
            
            # Evaluate
            probs = clf.predict_proba(X_test_s)[:, 1]
            y_pred = clf.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
            
            print(f"    Test Accuracy: {acc:.4f}")
            print(f"    Test AUROC:    {auc:.4f}")
            
            # Compute recommended threshold (median of predictions on val+test)
            all_probs = np.concatenate([
                clf.predict_proba(X_val_s)[:, 1],
                clf.predict_proba(X_test_s)[:, 1]
            ])
            recommended_threshold = np.percentile(all_probs, 50)
            
            # Save probe (with model identifier)
            model_name_safe = model_id.replace("/", "_")
            probe_name = f"{model_name_safe}_{feature_method}_{cfg['classifier']}"
            probe_dir = probes_dir / probe_name
            probe_dir.mkdir(exist_ok=True)
            
            # Save probe data
            probe_data = {
                "scaler": scaler,
                "classifier": clf,
                "threshold": thr,
                "recommended_threshold": recommended_threshold,
            }
            
            probe_pkl_path = probe_dir / "probe.pkl"
            with open(probe_pkl_path, "wb") as f:
                pickle.dump(probe_data, f)
            
            # Save metadata
            probe_metadata = {
                "model_id": model_id,
                "model_family": family,
                "feature_method": feature_method,
                "classifier": cfg["classifier"],
                "test_accuracy": float(acc),
                "test_auc": float(auc),
                "feature_dim": int(X.shape[1]),
                "layers": cfg["layers"],
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "semantic_entropy_threshold": float(thr),
                "recommended_threshold": float(recommended_threshold),
                "M_samples": cfg["M_samples"],
                "label_mode": cfg["label_mode"],
            }
            
            probe_json_path = probe_dir / "probe_metadata.json"
            with open(probe_json_path, "w") as f:
                json.dump(probe_metadata, f, indent=2)
            
            print(f"    ✅ Probe saved to: {probe_dir}/")
            
            all_results.append({
                "model_id": model_id,
                "model_family": family,
                "feature_method": feature_method,
                "classifier": cfg["classifier"],
                "test_accuracy": acc,
                "test_auc": auc,
                "feature_dim": X.shape[1],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "recommended_threshold": recommended_threshold,
                "probe_path": str(probe_dir),
            })
        
        training_time = time.time() - training_start
        print(f"\n✅ Training completed for {model_id} in {training_time/60:.1f} minutes ({training_time:.0f}s)")
    
    # Final summary across all models
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY (All Models)")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["model_id", "test_accuracy"], ascending=[True, False])
    
    print(results_df.to_string(index=False))
    
    # Save results
    results_file = output_dir / "probe_results.json"
    results_df.to_json(results_file, indent=2, orient="records")
    print(f"\n✅ Results saved to {results_file}")
    
    csv_file = output_dir / "probe_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"✅ Results saved to {csv_file}")
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"TOTAL TIME:           {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"Completed at:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return results_df

if __name__ == "__main__":
    results = run_experiment()

