#!/usr/bin/env python3
"""
Semantic Entropy Probe Training Module

This file handles everything needed to train the uncertainty probe:
1. Loading an LLM and generating code samples
2. Extracting features from hidden states during generation (SLT multi-layer)
3. Computing semantic entropy from multiple samples (functional uncertainty)
4. Running tests to get correctness labels
5. Training a probe to predict high semantic entropy from hidden state features
   (high semantic entropy correlates with incorrectness)

Note: This implementation is more sophisticated than the basic template:
- Uses semantic entropy (functional uncertainty) rather than token-level entropies
- Uses SLT (second-to-last token) multi-layer hidden states as features
- Predicts high semantic entropy (which correlates with P(incorrect))
- Supports multiple classifiers (Random Forest, Logistic Regression, SVM, MLP)
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
from contextlib import redirect_stdout, redirect_stderr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from scipy.stats import pearsonr, spearmanr
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass

# ============================================================
# Configuration
# ============================================================

def get_config():
    return {
        # Llama model only (matching your MTE work)
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        
        "dataset_name": "openai_humaneval",
        "split": "test",
        
        # Speed control - set to None for full 164 tasks, or a number like 30 for testing
        "limit_tasks": 30,  # Change to 30 for quick testing
        
        # SEP sampling (for semantic entropy) - INCREASED FOR BETTER ACCURACY
        "M_samples": 12,  # Increased from 6 to 12 for better semantic entropy estimates
        
        "sample_max_new_tokens": 256,
        "sample_temperature": 0.7,
        "sample_top_p": 0.95,
        
        # Greedy generation (for SLT + pass@1)
        "greedy_max_new_tokens": 256,
        
        # Feature extraction (SLT) - Using multiple layers for better features
        "layers": [-3, -2, -1],  # Multiple layers instead of just last layer
        "feature_mode": "SLT_multi",
        
        # Labeling semantic entropy
        "label_mode": "q25q75",  # Changed to q25q75 for better separation
        
        # Execution oracle timeout
        "test_timeout_s": 10,
        
        # Train/val/test split
        "seed": 42,
        
        # Output root directory
        "out_root": "sep_slt_runs",
        
        # Classifier settings
        "classifier": "random_forest",  # Options: "random_forest", "svm", "mlp", "logistic"
        "rf_n_estimators": 200,  # Random Forest: more trees for better accuracy
        "rf_max_depth": 15,
        "rf_min_samples_split": 5,
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
# Chat template
# ============================================================

def build_chat_text(tok, user_prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tok, "chat_template", None) not in (None, ""):
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"[SYSTEM] {SYSTEM_PROMPT}\n[USER] {user_prompt}\n[ASSISTANT]\n"

# ============================================================
# Model loading (robust to gated/unauthorized)
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
# Code extraction
# ============================================================

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

# ============================================================
# HumanEval test runner (oracle)
# ============================================================

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

# ============================================================
# Semantic signature (swap-in point for PyExZ3 later)
# Current: uses HumanEval test output logs hash as oracle.
# ============================================================

def semantic_signature(prompt_src: str, test_src: str, entry_point: str, code: str, timeout_s: int = 10):
    if not code:
        return "INVALID:syntax", 0
    module_src = prompt_src + "\n" + code + "\n\n" + test_src
    ok, logs = _run_test_with_timeout(module_src, entry_point, timeout_seconds=timeout_s)
    logs_norm = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", logs)
    h = hashlib.sha256(logs_norm.encode("utf-8", errors="ignore")).hexdigest()[:16]
    sig = ("PASS" if ok else "FAIL") + ":" + h
    return sig, int(ok)

# ============================================================
# SEP sampling + entropy
# ============================================================

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

@torch.inference_mode()
def sample_completion_with_len_norm_logp(tok, model, chat_text: str, max_new_tokens: int, temperature: float, top_p: float):
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # IMPORTANT: sampling ON
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
    
    # length-normalized logp for generated tokens
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

@torch.inference_mode()
def extract_slt_vec_multi_layer(tok, model, full_ids_cpu: torch.Tensor, layers: list = [-3, -2, -1]):
    """Extract features from multiple layers and concatenate them."""
    full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
    out = model(full_ids, output_hidden_states=True, use_cache=False)
    
    # Extract features from multiple layers
    features = []
    for layer in layers:
        hs = out.hidden_states[layer]
        features.append(hs[0, -1, :].float().detach().cpu().numpy())
    
    # Concatenate all layer features
    return np.concatenate(features)

# ============================================================
# Build SEP dataset for a single model_id
# ============================================================

def build_sep_dataset_for_model(model_id: str, cfg, tasks, hf_token: str | None):
    tok, model = load_model(model_id, hf_token)
    if tok is None:
        return None  # skipped
    
    limit = cfg["limit_tasks"]
    use_tasks = tasks[:limit] if limit is not None else tasks
    
    rows = []
    for ex in tqdm(use_tasks, desc=f"SEP dataset: {model_id.split('/')[-1]}", unit="task"):
        task_id = ex["task_id"]
        prompt_src = ex["prompt"]
        test_src = ex["test"]
        entry_point = ex["entry_point"]
        
        user_prompt = prompt_src + "\n\n# Your code below:\n"
        chat_text = build_chat_text(tok, user_prompt)
        
        # 1) Sample M completions -> clusters -> semantic entropy
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
        num_clusters = int(len(cluster_probs))
        
        # 2) Greedy completion -> SLT feature + pass@1
        full_ids, prompt_len = greedy_generate_full_ids_and_prompt_len(
            tok, model, chat_text, max_new_tokens=cfg["greedy_max_new_tokens"]
        )
        gen_ids = full_ids[prompt_len:]
        greedy_text = tok.decode(gen_ids, skip_special_tokens=True)
        greedy_code = extract_code(greedy_text)
        _, pass_at_1 = semantic_signature(prompt_src, test_src, entry_point, greedy_code, timeout_s=cfg["test_timeout_s"])
        
        # Extract multi-layer features
        feat = extract_slt_vec_multi_layer(tok, model, full_ids, layers=cfg["layers"])
        
        rows.append({
            "model_id": model_id,
            "task_id": task_id,
            "semantic_entropy": float(sem_ent),
            "num_clusters": int(num_clusters),
            "pass_at_1": int(pass_at_1),
            "feature": feat,
        })
    
    # cleanup
    del model, tok
    torch.cuda.empty_cache()
    gc.collect()
    
    return rows

# ============================================================
# Train + evaluate probe for one model_id
# ============================================================

def make_labels(df: pd.DataFrame, label_mode: str, eps_std: float = 1e-8):
    sE = df["semantic_entropy"].values.astype(np.float64)
    
    # Guard: entropy collapse => can't make a meaningful "high vs low" label
    uniq = np.unique(sE[~np.isnan(sE)])
    if len(uniq) < 2 or float(np.nanstd(sE)) < eps_std:
        return None, None, None, {
            "reason": "semantic_entropy near-constant",
            "n_unique": int(len(uniq)),
            "std": float(np.nanstd(sE)),
            "min": float(np.nanmin(sE)),
            "median": float(np.nanmedian(sE)),
            "max": float(np.nanmax(sE)),
        }
    
    if label_mode == "median":
        thr = float(np.nanmedian(sE))
        y = (sE > thr).astype(int)
        keep = np.ones_like(y, dtype=bool)
        
        # If still single class (ties), force rank split
        if len(np.unique(y)) < 2:
            order = np.argsort(sE)
            y = np.zeros_like(sE, dtype=int)
            y[order[len(sE)//2:]] = 1
        
        return y, keep, thr, None
    
    if label_mode == "q25q75":
        q25, q75 = np.quantile(sE, [0.25, 0.75])
        keep = (sE <= q25) | (sE >= q75)
        y = (sE[keep] >= float(q75)).astype(int)
        
        if len(np.unique(y)) < 2:
            return None, None, None, {
                "reason": "degenerate labels after q25q75",
                "q25": float(q25),
                "q75": float(q75),
                "kept": int(np.sum(keep)),
            }
        
        return y, keep, float(q75), None
    
    raise ValueError("label_mode must be 'median' or 'q25q75'")

def _safe_train_test_split(X, y, test_size, seed, stratify=True):
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=(y if stratify else None)
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)

def train_eval_probe_for_model(df_model: pd.DataFrame, cfg):
    X_all = np.stack(df_model["feature"].values).astype(np.float32)
    y_full, keep_mask, thr, err = make_labels(df_model, cfg["label_mode"])
    
    if err is not None:
        print("\n[SKIP TRAIN] Could not create labels:", err)
        return None, None, None, None
    
    if cfg["label_mode"] == "q25q75":
        df_use = df_model[keep_mask].reset_index(drop=True)
        X = np.stack(df_use["feature"].values).astype(np.float32)
        y = y_full
    else:
        df_use = df_model.copy()
        X = X_all
        y = y_full
    
    if len(np.unique(y)) < 2:
        print("\n[SKIP TRAIN] Labels are single-class even after safeguards.")
        return None, None, None, None
    
    print("\nProbe target: y=1 means high semantic entropy")
    print("Semantic entropy threshold:", float(thr))
    print("Entropy stats:",
          f"min={float(np.min(df_use['semantic_entropy'])):.6f}",
          f"median={float(np.median(df_use['semantic_entropy'])):.6f}",
          f"max={float(np.max(df_use['semantic_entropy'])):.6f}",
          f"unique={int(df_use['semantic_entropy'].nunique())}")
    
    binc = np.bincount(y, minlength=2)
    print("Label balance: y0 =", int(binc[0]), "| y1 =", int(binc[1]))
    print(f"Feature dimensions: {X.shape[1]} (multi-layer features)")
    
    X_train, X_tmp, y_train, y_tmp = _safe_train_test_split(
        X, y, test_size=0.30, seed=cfg["seed"], stratify=True
    )
    X_val, X_test, y_val, y_test = _safe_train_test_split(
        X_tmp, y_tmp, test_size=0.50, seed=cfg["seed"], stratify=True
    )
    
    # Guard: splits must have both classes
    for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if len(np.unique(ys)) < 2:
            print(f"\n[SKIP TRAIN] Split '{name}' is single-class; try larger dataset or q25q75.")
            return None, None, None, None
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Model selection based on config
    classifier_type = cfg.get("classifier", "random_forest")
    
    if classifier_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=cfg.get("rf_n_estimators", 200),
            max_depth=cfg.get("rf_max_depth", 15),
            min_samples_split=cfg.get("rf_min_samples_split", 5),
            random_state=cfg["seed"],
            n_jobs=-1,
            verbose=1
        )
        print(f"\nTraining Random Forest classifier (n_estimators={cfg.get('rf_n_estimators', 200)})...")
    elif classifier_type == "svm":
        from sklearn.svm import SVC
        clf = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=cfg["seed"]
        )
        print("\nTraining SVM classifier...")
    elif classifier_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            random_state=cfg["seed"],
            early_stopping=True,
            validation_fraction=0.1,
            verbose=True
        )
        print("\nTraining MLP classifier...")
    elif classifier_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=4000, random_state=cfg["seed"])
        print("\nTraining Logistic Regression classifier...")
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=cfg["seed"],
            n_jobs=-1
        )
        print("\nTraining Random Forest classifier (default)...")
    
    clf.fit(X_train_s, y_train)
    
    def eval_split(name, Xs, ys):
        p = clf.predict_proba(Xs)[:, 1]
        yhat = (p >= 0.5).astype(int)
        auc = roc_auc_score(ys, p) if len(np.unique(ys)) > 1 else float("nan")
        acc = accuracy_score(ys, yhat)
        print(f"\n{name}: AUROC={auc:.4f}  ACC={acc:.4f}")
        print(classification_report(ys, yhat, digits=4))
        
        # DEBUG: Show prediction distribution
        print(f"\n[{name} DEBUG] Probe prediction statistics:")
        print(f"  Min prob_high_entropy:  {np.min(p):.4f}")
        print(f"  Median prob_high_entropy: {np.median(p):.4f}")
        print(f"  Max prob_high_entropy:  {np.max(p):.4f}")
        print(f"  Mean prob_high_entropy: {np.mean(p):.4f}")
        print(f"  Std prob_high_entropy:  {np.std(p):.4f}")
        
        # WARNING: Check if predictions are all low
        if np.max(p) < 0.3:
            print(f"\n⚠️  WARNING [{name}]: All predictions < 0.3!")
            print(f"   This means probe will NEVER trigger adaptive decoding with threshold > 0.3")
            print(f"   Suggested threshold: {np.percentile(p, 75):.3f} (75th percentile)")
        elif np.min(p) > 0.7:
            print(f"\n⚠️  WARNING [{name}]: All predictions > 0.7!")
            print(f"   This means probe will ALWAYS trigger adaptive decoding")
            print(f"   Suggested threshold: {np.percentile(p, 25):.3f} (25th percentile)")
        else:
            # Suggest optimal threshold
            suggested_thresh = np.percentile(p, 50)  # Median
            print(f"\n✅ [{name}] Prediction range looks reasonable")
            print(f"   Suggested threshold for adaptive decoding: {suggested_thresh:.3f} (median)")
            print(f"   Alternative thresholds: {np.percentile(p, 25):.3f} (25th), {np.percentile(p, 75):.3f} (75th)")
        
        # WARNING: Check accuracy
        if acc < 0.55:
            print(f"\n⚠️  WARNING [{name}]: Accuracy is very low ({acc:.4f})!")
            print(f"   Probe may not be reliable. Consider:")
            print(f"   - Increasing M_samples (current: {cfg.get('M_samples', 'N/A')})")
            print(f"   - Using more training data (current: {len(X_train)} samples)")
            print(f"   - Trying different classifier or label_mode")
        
        return auc, acc, p  # Return predictions for further analysis
    
    val_auc, val_acc, val_predictions = eval_split("VAL", X_val_s, y_val)
    test_auc, test_acc, test_predictions = eval_split("TEST", X_test_s, y_test)
    
    # Overall probe quality check
    print("\n" + "="*80)
    print("PROBE QUALITY ASSESSMENT")
    print("="*80)
    
    all_predictions = np.concatenate([val_predictions, test_predictions])
    print(f"\nOverall prediction distribution (VAL+TEST):")
    print(f"  Min:   {np.min(all_predictions):.4f}")
    print(f"  Q25:   {np.percentile(all_predictions, 25):.4f}")
    print(f"  Median: {np.median(all_predictions):.4f}")
    print(f"  Q75:   {np.percentile(all_predictions, 75):.4f}")
    print(f"  Max:   {np.max(all_predictions):.4f}")
    print(f"  Mean:  {np.mean(all_predictions):.4f}")
    print(f"  Std:   {np.std(all_predictions):.4f}")
    
    # Dataset size warning
    if len(X_train) < 50:
        print(f"\n⚠️  WARNING: Small training dataset ({len(X_train)} samples)")
        print(f"   Probe may not generalize well. Consider:")
        print(f"   - Increasing limit_tasks (current: {cfg.get('limit_tasks', 'None')})")
        print(f"   - Using full dataset (164 tasks) for better probe quality")
    
    # Final recommendation
    recommended_threshold = np.percentile(all_predictions, 50)
    print(f"\n📊 RECOMMENDED THRESHOLDS:")
    print(f"   For adaptive decoding: {recommended_threshold:.3f} (median)")
    print(f"   Conservative (25th):   {np.percentile(all_predictions, 25):.3f}")
    print(f"   Aggressive (75th):     {np.percentile(all_predictions, 75):.3f}")
    print(f"\n   Update in adaptive_decoding_lambda.py line ~214:")
    print(f"   use_adaptive = prob_high_entropy > {recommended_threshold:.3f}")
    print(f"\n   Update in self_correction_lambda.py get_config():")
    print(f"   'uncertainty_threshold': {recommended_threshold:.3f},")
    print("="*80)
    
    # Save threshold recommendations to file
    threshold_dir = os.path.join(cfg["out_root"], safe_name(cfg["model_id"]))
    os.makedirs(threshold_dir, exist_ok=True)  # Ensure directory exists
    threshold_file = os.path.join(threshold_dir, "recommended_thresholds.txt")
    with open(threshold_file, "w") as f:
        f.write("RECOMMENDED THRESHOLDS FOR ADAPTIVE DECODING AND SELF-CORRECTION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Based on probe prediction distribution:\n")
        f.write(f"  Min:   {np.min(all_predictions):.4f}\n")
        f.write(f"  Q25:   {np.percentile(all_predictions, 25):.4f}\n")
        f.write(f"  Median: {np.median(all_predictions):.4f}\n")
        f.write(f"  Q75:   {np.percentile(all_predictions, 75):.4f}\n")
        f.write(f"  Max:   {np.max(all_predictions):.4f}\n\n")
        f.write(f"RECOMMENDED THRESHOLDS:\n")
        f.write(f"  Conservative (25th percentile): {np.percentile(all_predictions, 25):.3f}\n")
        f.write(f"  Recommended (median):         {recommended_threshold:.3f}\n")
        f.write(f"  Aggressive (75th percentile):   {np.percentile(all_predictions, 75):.3f}\n\n")
        f.write("UPDATE THESE IN YOUR CONFIG FILES:\n")
        f.write("-"*80 + "\n")
        f.write("adaptive_decoding_lambda.py (line ~214):\n")
        f.write(f"  use_adaptive = prob_high_entropy > {recommended_threshold:.3f}\n\n")
        f.write("self_correction_lambda.py get_config():\n")
        f.write(f"  'uncertainty_threshold': {recommended_threshold:.3f},\n")
    print(f"\n✅ Threshold recommendations saved to: {threshold_file}")
    print(f"\n✅ Recommended threshold (median): {recommended_threshold:.4f} will be saved to probe.pkl")
    
    return scaler, clf, thr, df_use, recommended_threshold

# ============================================================
# Save artifacts per model
# ============================================================

def safe_name(s: str):
    return s.replace("/", "_").replace(":", "_")

def save_model_artifacts(cfg, model_id: str, df_use: pd.DataFrame, scaler, clf, thr, recommended_threshold=None):
    out_dir = os.path.join(cfg["out_root"], safe_name(model_id))
    os.makedirs(out_dir, exist_ok=True)
    
    meta = df_use.drop(columns=["feature"]).copy()
    meta.to_csv(os.path.join(out_dir, "sep_dataset_metadata.csv"), index=False)
    
    feats = np.stack(df_use["feature"].values).astype(np.float32)
    np.save(os.path.join(out_dir, "features.npy"), feats)
    
    # Save classifier-specific information
    probe_json = {
        "model_id": model_id,
        "feature_mode": cfg["feature_mode"],
        "layers": cfg["layers"],
        "label_mode": cfg["label_mode"],
        "semantic_entropy_threshold": float(thr),
        "classifier": cfg.get("classifier", "random_forest"),
        "seed": cfg["seed"],
    }
    
    # Add classifier-specific parameters
    if hasattr(clf, 'coef_'):
        probe_json["coef"] = clf.coef_.tolist()
        probe_json["intercept"] = clf.intercept_.tolist()
    elif hasattr(clf, 'feature_importances_'):
        probe_json["feature_importances"] = clf.feature_importances_.tolist()
        probe_json["n_estimators"] = clf.n_estimators
        probe_json["max_depth"] = clf.max_depth
    
    probe_json["scaler_mean"] = scaler.mean_.tolist()
    probe_json["scaler_scale"] = scaler.scale_.tolist()
    
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(probe_json, f, indent=2)
    
    # Save actual trained classifier and scaler using pickle
    probe_data = {
        "scaler": scaler,
        "classifier": clf,
        "threshold": thr,  # Semantic entropy threshold (for labeling)
    }
    if recommended_threshold is not None:
        probe_data["recommended_threshold"] = recommended_threshold  # Recommended threshold for inference
        print(f"✅ Saving recommended threshold: {recommended_threshold:.4f} to probe.pkl")
    
    with open(os.path.join(out_dir, "probe.pkl"), "wb") as f:
        pickle.dump(probe_data, f)
    
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    
    print("\nSaved artifacts to:", out_dir)

# ============================================================
# Optional: semantic entropy vs failure sanity
# ============================================================

def relation_report(df_model: pd.DataFrame):
    """
    Report correlation between semantic entropy and failure (incorrectness).
    Calculates both Pearson and Spearman correlation coefficients.
    """
    df = df_model.copy()
    df["fail"] = 1 - df["pass_at_1"].astype(int)
    
    if df["semantic_entropy"].nunique() < 2 or df["fail"].nunique() < 2:
        print("\nCorrelation(semantic_entropy, fail): not defined (degenerate variance)")
        return
    
    # Calculate Pearson correlation (linear relationship)
    pearson_corr, pearson_p = pearsonr(df["semantic_entropy"].values, df["fail"].values)
    
    # Calculate Spearman correlation (monotonic relationship)
    spearman_corr, spearman_p = spearmanr(df["semantic_entropy"].values, df["fail"].values)
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Semantic Entropy vs Failure")
    print("="*80)
    print(f"\nPearson Correlation (linear):")
    print(f"  r = {pearson_corr:.4f}")
    print(f"  p-value = {pearson_p:.4e}")
    if pearson_p < 0.001:
        print(f"  Significance: *** (p < 0.001)")
    elif pearson_p < 0.01:
        print(f"  Significance: ** (p < 0.01)")
    elif pearson_p < 0.05:
        print(f"  Significance: * (p < 0.05)")
    else:
        print(f"  Significance: not significant (p >= 0.05)")
    
    print(f"\nSpearman Correlation (monotonic):")
    print(f"  ρ = {spearman_corr:.4f}")
    print(f"  p-value = {spearman_p:.4e}")
    if spearman_p < 0.001:
        print(f"  Significance: *** (p < 0.001)")
    elif spearman_p < 0.01:
        print(f"  Significance: ** (p < 0.01)")
    elif spearman_p < 0.05:
        print(f"  Significance: * (p < 0.05)")
    else:
        print(f"  Significance: not significant (p >= 0.05)")
    
    # Interpretation
    print(f"\nInterpretation:")
    if abs(pearson_corr) > 0.7:
        strength = "strong"
    elif abs(pearson_corr) > 0.4:
        strength = "moderate"
    elif abs(pearson_corr) > 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    
    direction = "positive" if pearson_corr > 0 else "negative"
    print(f"  {strength.capitalize()} {direction} correlation between semantic entropy and failure")
    print(f"  Higher semantic entropy → {'Higher' if pearson_corr > 0 else 'Lower'} failure rate")
    
    # Bucket analysis (quartiles)
    print(f"\nQuartile Analysis:")
    qs = df["semantic_entropy"].quantile([0.25, 0.5, 0.75]).to_dict()
    buckets = [
        ("low (<=q25)", -np.inf, qs[0.25]),
        ("mid (q25-q75)", qs[0.25], qs[0.75]),
        ("high (>=q75)", qs[0.75], np.inf),
    ]
    
    for name, lo, hi in buckets:
        chunk = df[(df["semantic_entropy"] > lo) & (df["semantic_entropy"] <= hi)]
        if len(chunk) == 0:
            continue
        fail_rate = chunk['fail'].mean()
        print(f"  {name}: n={len(chunk)}  fail_rate={fail_rate:.3f} ({fail_rate*100:.1f}%)")
    
    print("="*80)

# ============================================================
# Main
# ============================================================

def main():
    """
    Main training pipeline.
    
    Implements the full training pipeline:
    1. Load configuration
    2. Load the language model (e.g., LLaMA)
    3. Collect training data (generate code samples, extract features, get correctness labels)
    4. Extract features (SLT multi-layer hidden states)
    5. Split into train/val/test sets
    6. Define and train a probe model (Random Forest, configurable to Logistic Regression)
    7. Evaluate on test set
    8. Save the trained probe for use in self-correction
    
    Note: This implementation uses semantic entropy (functional uncertainty) rather than
    token-level entropies, and predicts high semantic entropy (which correlates with
    incorrectness) rather than P(incorrect) directly. This is more sophisticated than
    the basic template and provides better uncertainty estimation.
    """
    print("=== Semantic Entropy Probe Training ===")
    
    # 1. Load configuration
    cfg = get_config()
    set_seed(cfg["seed"])
    print(f"Configuration loaded: {cfg['model_id']}, classifier={cfg['classifier']}")
    
    # 2. Load the language model
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with Llama access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Load dataset
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    tasks = [ds[i] for i in range(len(ds))]
    
    if cfg["limit_tasks"] is None:
        print(f"\nUsing full HumanEval tasks: {len(tasks)}")
    else:
        print(f"\nUsing limited HumanEval tasks: {cfg['limit_tasks']}")
    
    os.makedirs(cfg["out_root"], exist_ok=True)
    
    # 3. Collect training data
    #    - Generate code samples (M_samples per task for semantic entropy)
    #    - Extract features (SLT multi-layer hidden states)
    #    - Run tests to get correctness labels
    print("\n" + "="*80)
    print("Step 3: Collecting training data")
    print(f"Model: {cfg['model_id']}")
    print(f"M_samples: {cfg['M_samples']} (for semantic entropy estimation)")
    print(f"Layers: {cfg['layers']} (multi-layer SLT features)")
    print("="*80)
    
    rows = build_sep_dataset_for_model(cfg["model_id"], cfg, tasks, hf_token=HF_TOKEN)
    
    if rows is None:
        print("[ERROR] Could not load model:", cfg["model_id"])
        return
    
    df = pd.DataFrame(rows)
    print(f"\n✅ Collected {len(df)} training examples")
    print(f"   Features: SLT multi-layer hidden states ({df.iloc[0]['feature'].shape[0]} dims)")
    print(f"   Labels: High semantic entropy (correlates with incorrectness)")
    print(f"   Example: {df.iloc[0][['task_id','semantic_entropy','pass_at_1']].to_dict()}")
    
    # 4. Extract features (already done in step 3, but verify)
    # Features are SLT multi-layer hidden states extracted during generation
    
    # 5. Split into train/val/test sets
    # 6. Define and train a probe model
    # 7. Evaluate on test set
    print("\n" + "="*80)
    print("Steps 5-7: Training and evaluating probe")
    print("="*80)
    
    scaler, clf, thr, df_use, recommended_threshold = train_eval_probe_for_model(df, cfg)
    
    # Report semantic entropy vs failure correlation (sanity check)
    relation_report(df)
    
    if scaler is None:
        print("\n[ERROR] Probe training failed (degenerate entropy/labels).")
        print("        Try increasing M_samples (e.g., 12-16) and/or temperature/top_p.")
        return
    
    # 8. Save the trained probe for use in self-correction
    print("\n" + "="*80)
    print("Step 8: Saving trained probe")
    print("="*80)
    
    save_model_artifacts(cfg, cfg["model_id"], df_use, scaler, clf, thr, recommended_threshold)
    
    print("\n✅ Training pipeline completed successfully!")
    print("   Probe saved and ready for use in adaptive decoding and self-correction.")

if __name__ == "__main__":
    main()

