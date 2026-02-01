#!/usr/bin/env python3
"""
Threshold Tuning Analysis for Probe Experiments

This script tests different threshold values (0.3, 0.4, 0.5, 0.6, 0.7) on the test set
for each trained probe. It helps understand:
1. How each threshold performs on the test set
2. When threshold tuning would trigger corrections
3. Precision, recall, F1, and accuracy at each threshold
4. Trigger rates (what % of examples would trigger corrections)

Usage:
    python thresh_tune.py
"""

import os
import json
import pickle
import ast
import re
import io
import signal
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score, confusion_matrix
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
from getpass import getpass
from contextlib import redirect_stdout, redirect_stderr

# Try to import matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  matplotlib/seaborn not available. Skipping visualizations.")

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent
SAVED_PROBES_DIR = SCRIPT_DIR / "saved_probes"
OUTPUT_DIR = SCRIPT_DIR / "threshold_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Same config as train_probes.py
EXPERIMENT_CONFIG = {
    "seed": 42,
    "label_mode": "median",  # Should match what was used in training
}

# Threshold values to test
THRESHOLDS_TO_TEST = [0.3, 0.4, 0.5, 0.6, 0.7]

# ============================================================
# Load Functions
# ============================================================

def load_saved_probe(probe_dir: Path):
    """Load a saved probe from disk."""
    probe_pkl_path = probe_dir / "probe.pkl"
    probe_json_path = probe_dir / "probe_metadata.json"
    
    if not probe_pkl_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_pkl_path}")
    
    with open(probe_pkl_path, "rb") as f:
        probe_data = pickle.load(f)
    
    scaler = probe_data["scaler"]
    clf = probe_data["classifier"]
    threshold = probe_data.get("threshold", None)
    recommended_threshold = probe_data.get("recommended_threshold", 0.5)
    
    metadata = None
    if probe_json_path.exists():
        with open(probe_json_path, "r") as f:
            metadata = json.load(f)
    
    return scaler, clf, threshold, recommended_threshold, metadata

def find_all_probes(probes_dir: Path):
    """Find all saved probes."""
    probes = []
    if not probes_dir.exists():
        print(f"❌ Probes directory not found: {probes_dir}")
        return probes
    
    for probe_dir in probes_dir.iterdir():
        if probe_dir.is_dir():
            probe_pkl = probe_dir / "probe.pkl"
            if probe_pkl.exists():
                probes.append(probe_dir)
    
    return sorted(probes)

def find_dataset_files(dataset_dir: Path):
    """Find all dataset files (prefer pickle, fallback to CSV)."""
    datasets = {}
    
    # First, look for pickle files (preserve full arrays)
    for pkl_file in dataset_dir.glob("probe_dataset_*.pkl"):
        model_name = pkl_file.stem.replace("probe_dataset_", "")
        datasets[model_name] = ("pkl", pkl_file)
    
    # Then, look for CSV files (may have truncated arrays)
    for csv_file in dataset_dir.glob("probe_dataset_*.csv"):
        model_name = csv_file.stem.replace("probe_dataset_", "")
        if model_name not in datasets:  # Only add if no pickle file exists
            datasets[model_name] = ("csv", csv_file)
    
    return datasets

# ============================================================
# Dataset Recreation (Same Split as Training)
# ============================================================

def parse_array_string(arr_str):
    """
    Parse a string representation of a numpy array back into a numpy array.
    Handles formats like: '[ 5.5        0.1953125 -2.765625  ... ]'
    or full arrays saved by pandas.
    """
    # If already an array or list, return as numpy array
    if isinstance(arr_str, np.ndarray):
        return arr_str.astype(np.float32)
    if isinstance(arr_str, list):
        return np.array(arr_str, dtype=np.float32)
    
    arr_str_orig = str(arr_str).strip()
    arr_str = arr_str_orig
    
    # Method 1: Try ast.literal_eval (safest, handles full arrays as Python lists)
    try:
        arr = ast.literal_eval(arr_str)
        if isinstance(arr, list):
            result = np.array(arr, dtype=np.float32)
            if len(result) > 100:  # Sanity check - should be thousands of features
                return result
    except (ValueError, SyntaxError):
        pass
    
    # Method 2: Try eval (handles numpy array strings and numpy array() calls)
    try:
        # Replace numpy array syntax if present
        arr_str_eval = arr_str.replace('array(', '').replace(')', '')
        arr = eval(arr_str_eval)
        if isinstance(arr, (list, np.ndarray)):
            result = np.array(arr, dtype=np.float32)
            if len(result) > 100:
                return result
    except:
        pass
    
    # Method 3: Try parsing as space-separated numbers (handles full arrays without brackets)
    # This handles cases where the CSV has the full array as space-separated numbers
    try:
        # Remove brackets and newlines
        cleaned = arr_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
        
        # If it contains '...', the array is truncated in the CSV - we can't recover it
        if '...' in cleaned:
            raise ValueError(
                f"Array is truncated in CSV (contains '...'). "
                f"Expected ~6144 or ~9216 features but CSV only has partial data. "
                f"First 500 chars: {arr_str_orig[:500]}"
            )
        
        # Parse all numbers
        nums = []
        for token in cleaned.split():
            token = token.strip()
            if token:
                try:
                    nums.append(float(token))
                except ValueError:
                    pass
        
        if nums and len(nums) > 100:  # Should have thousands of features
            return np.array(nums, dtype=np.float32)
        elif nums:
            raise ValueError(
                f"Parsed only {len(nums)} features, expected thousands. "
                f"Array appears to be truncated. First 500 chars: {arr_str_orig[:500]}"
            )
    except ValueError:
        raise  # Re-raise our custom ValueError
    except Exception:
        pass
    
    # If all methods fail, raise error with context
    raise ValueError(
        f"Could not parse array string.\n"
        f"Type: {type(arr_str)}\n"
        f"String length: {len(arr_str_orig)}\n"
        f"First 500 chars: {arr_str_orig[:500]}\n"
        f"Last 200 chars: {arr_str_orig[-200:] if len(arr_str_orig) > 200 else arr_str_orig}\n"
        f"Contains '...': {'...' in arr_str_orig}"
    )

def load_test_set_task_ids(split_dir: Path):
    """Load test set task IDs from DatasetSplit folder."""
    test_tasks_file = split_dir / "test_tasks.csv"
    if not test_tasks_file.exists():
        return None
    
    test_df = pd.read_csv(test_tasks_file)
    return set(test_df["task_id"].tolist())

def regenerate_test_set_features(model_id: str, test_task_ids: set, feature_method: str, 
                                 layers: list, hf_token: str = None):
    """
    Regenerate features for test set tasks only.
    This is much faster than regenerating all 164 tasks.
    """
    print(f"\n  Regenerating features for {len(test_task_ids)} test set tasks...")
    print(f"  This requires loading the model (one-time cost)...")
    
    # Load model
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
    print(f"  ✅ Model loaded")
    
    # Load HumanEval dataset
    ds = load_dataset("openai_humaneval")["test"]
    tasks_dict = {ex["task_id"]: ex for ex in ds}
    
    # Determine model family for prompt building
    if "llama" in model_id.lower():
        family = "llama"
    elif "qwen" in model_id.lower():
        family = "qwen-coder-instruct"
    elif "deepseek" in model_id.lower():
        family = "deepseek"
    else:
        family = "llama"  # default
    
    # Extract features for test set tasks
    test_features = {}
    
    def build_chat_text_simple(prompt: str, family: str):
        """Simplified chat text builder."""
        if family in {"llama", "deepseek", "qwen-coder-instruct"}:
            system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
            user = prompt + "\n\n# Your code below:\n"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            if getattr(tok, "chat_template", None) not in (None, ""):
                return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return f"[SYSTEM] {system}\n[USER] {user}\n[ASSISTANT]\n"
        return prompt + "\n# Your code below:\n"
    
    @torch.inference_mode()
    def greedy_generate_ids(tok, model, chat_text: str, max_new_tokens: int = 256):
        """Generate greedy completion and return full token IDs."""
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
    def extract_features_simple(tok, model, full_ids: torch.Tensor, prompt_len: int, 
                                layers: list, method: str):
        """Extract features using specified method."""
        full_ids = full_ids.unsqueeze(0).to(model.device)
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
            
            if token_idx < 0:
                token_idx = hs.shape[1] + token_idx
            if token_idx >= hs.shape[1]:
                token_idx = hs.shape[1] - 1
            if token_idx < 0:
                token_idx = 0
            
            features.append(hs[0, token_idx, :].float().detach().cpu().numpy())
        
        return np.concatenate(features)
    
    # Process each test task
    for task_id in tqdm(test_task_ids, desc="  Extracting features"):
        if task_id not in tasks_dict:
            print(f"    ⚠️  Task {task_id} not found in dataset, skipping")
            continue
        
        ex = tasks_dict[task_id]
        prompt_src = ex["prompt"]
        
        # Build prompt
        chat_text = build_chat_text_simple(prompt_src, family)
        
        # Generate greedy completion
        full_ids, prompt_len = greedy_generate_ids(tok, model, chat_text, max_new_tokens=256)
        
        # Extract features
        feat = extract_features_simple(tok, model, full_ids, prompt_len, layers, feature_method)
        test_features[task_id] = feat
    
    # Cleanup
    del model, tok
    torch.cuda.empty_cache()
    
    print(f"  ✅ Extracted features for {len(test_features)} test tasks")
    return test_features

def recreate_test_split(df, feature_method: str, seed: int = 42):
    """
    Recreate the exact same train/val/test split used during training.
    
    This uses the same random_state and stratification as train_probes.py
    """
    # Extract features and labels (same as training)
    if feature_method not in df.columns:
        raise ValueError(f"Feature method '{feature_method}' not found in dataset")
    
    # Parse array strings back into numpy arrays
    feature_arrays = []
    for val in df[feature_method].values:
        arr = parse_array_string(val)
        feature_arrays.append(arr)
    
    X = np.stack(feature_arrays).astype(np.float32)
    
    # Recreate labels (same as training)
    semantic_entropy = df["semantic_entropy"].values
    label_mode = EXPERIMENT_CONFIG["label_mode"]
    
    if label_mode == "median":
        thr = float(np.nanmedian(semantic_entropy))
        y = (semantic_entropy > thr).astype(int)
        keep = np.ones_like(y, dtype=bool)
    else:  # q25q75
        q25, q75 = np.quantile(semantic_entropy, [0.25, 0.75])
        keep = (semantic_entropy <= q25) | (semantic_entropy >= q75)
        y = (semantic_entropy[keep] >= float(q75)).astype(int)
        X = X[keep]
        df = df[keep].reset_index(drop=True)
    
    # Same split as training: 70% train, 15% val, 15% test
    # We need to use indices to maintain the same split
    indices = np.arange(len(X))
    indices_train, indices_tmp, y_train, y_tmp = train_test_split(
        indices, y, test_size=0.30, random_state=seed, stratify=y
    )
    indices_val, indices_test, y_val, y_test = train_test_split(
        indices_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    
    # Extract test set features and labels
    X_test = X[indices_test]
    y_test = y_test
    
    return {
        "X_test": X_test,
        "y_test": y_test,
        "indices_test": indices_test,
    }

# ============================================================
# Threshold Analysis
# ============================================================

def evaluate_at_threshold(y_true, y_probs, threshold: float):
    """Evaluate metrics at a specific threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    
    # Handle edge cases
    if len(np.unique(y_pred)) == 1:
        # All predictions are the same
        if y_pred[0] == 0:
            precision = 0.0 if np.sum(y_true) > 0 else 1.0
            recall = 0.0
        else:
            precision = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
            recall = 1.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
    
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate trigger rate (what % of examples would trigger correction)
    trigger_rate = np.mean(y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "trigger_rate": trigger_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

def analyze_probe_thresholds(probe_dir: Path, df: pd.DataFrame, split_dir: Path = None, 
                             hf_token: str = None, regenerate_features: bool = True):
    """
    Analyze thresholds for a single probe.
    
    Uses DatasetSplit to identify test set tasks, then regenerates features only for those tasks.
    This avoids needing full feature arrays from CSV.
    
    Returns:
        dict with threshold analysis results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing probe: {probe_dir.name}")
    print(f"{'='*80}")
    
    # Load probe
    try:
        scaler, clf, threshold, recommended_threshold, metadata = load_saved_probe(probe_dir)
    except Exception as e:
        print(f"❌ Failed to load probe: {e}")
        return None
    
    if metadata is None:
        print("⚠️  No metadata found, cannot determine feature method")
        return None
    
    feature_method = metadata.get("feature_method")
    classifier_type = metadata.get("classifier")
    model_id = metadata.get("model_id", "unknown")
    layers = metadata.get("layers", [-3, -2, -1])
    
    print(f"Model: {model_id}")
    print(f"Feature Method: {feature_method}")
    print(f"Classifier: {classifier_type}")
    print(f"Current Recommended Threshold: {recommended_threshold:.4f}")
    
    # Get test set task IDs from DatasetSplit
    if split_dir is None:
        split_dir = SCRIPT_DIR / "DatasetSplit"
    
    test_task_ids = load_test_set_task_ids(split_dir)
    if test_task_ids is None:
        print(f"❌ Could not load test set task IDs from {split_dir}")
        print(f"   Falling back to recreating split from full dataset...")
        # Fallback to old method (will fail if CSV has truncated arrays)
        try:
            splits = recreate_test_split(df, feature_method, seed=EXPERIMENT_CONFIG["seed"])
            X_test = splits["X_test"]
            y_test = splits["y_test"]
        except Exception as e:
            print(f"❌ Failed: {e}")
            return None
    else:
        print(f"✅ Found {len(test_task_ids)} test set tasks from DatasetSplit")
        
        # Filter dataset to test set tasks and get semantic entropy (for labels)
        df_test = df[df["task_id"].isin(test_task_ids)].copy()
        if len(df_test) == 0:
            print(f"❌ No matching tasks found in dataset CSV")
            return None
        
        print(f"✅ Found {len(df_test)} test tasks in dataset")
        
        # Get semantic entropy for labels
        semantic_entropy = df_test["semantic_entropy"].values
        label_mode = EXPERIMENT_CONFIG["label_mode"]
        
        if label_mode == "median":
            thr = float(np.nanmedian(semantic_entropy))
            y_test = (semantic_entropy > thr).astype(int)
        else:  # q25q75
            q25, q75 = np.quantile(semantic_entropy, [0.25, 0.75])
            keep = (semantic_entropy <= q25) | (semantic_entropy >= q75)
            y_test = (semantic_entropy[keep] >= float(q75)).astype(int)
            df_test = df_test[keep].reset_index(drop=True)
            test_task_ids = set(df_test["task_id"].tolist())
        
        # Regenerate features for test set tasks only
        if regenerate_features:
            try:
                test_features_dict = regenerate_test_set_features(
                    model_id, test_task_ids, feature_method, layers, hf_token
                )
                
                # Build feature array in same order as df_test
                X_test_list = []
                y_test_filtered = []
                for idx, task_id in enumerate(df_test["task_id"]):
                    if task_id in test_features_dict:
                        X_test_list.append(test_features_dict[task_id])
                        y_test_filtered.append(y_test[idx])
                
                if len(X_test_list) == 0:
                    print(f"❌ No features extracted for test set")
                    return None
                
                X_test = np.stack(X_test_list).astype(np.float32)
                y_test = np.array(y_test_filtered)
                
            except Exception as e:
                print(f"❌ Failed to regenerate features: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # Try to use features from CSV (may be truncated)
            try:
                X_test = np.stack(df_test[feature_method].apply(parse_array_string).values).astype(np.float32)
            except Exception as e:
                print(f"❌ Cannot use CSV features (truncated): {e}")
                return None
    
    # Scale test features
    X_test_s = scaler.transform(X_test)
    
    # Get predictions
    y_probs = clf.predict_proba(X_test_s)[:, 1]
    
    print(f"\nTest Set Statistics:")
    print(f"  Size: {len(y_test)}")
    print(f"  Positive class (high entropy): {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    print(f"  Negative class (low entropy): {len(y_test) - np.sum(y_test)} ({(1-np.mean(y_test))*100:.1f}%)")
    print(f"\nPrediction Statistics:")
    print(f"  Min: {np.min(y_probs):.4f}")
    print(f"  Median: {np.median(y_probs):.4f}")
    print(f"  Max: {np.max(y_probs):.4f}")
    print(f"  Mean: {np.mean(y_probs):.4f}")
    print(f"  Std: {np.std(y_probs):.4f}")
    
    # Test each threshold
    print(f"\n{'─'*80}")
    print("THRESHOLD ANALYSIS (0.3, 0.4, 0.5, 0.6, 0.7)")
    print(f"{'─'*80}")
    
    threshold_results = []
    
    for thresh in THRESHOLDS_TO_TEST:
        result = evaluate_at_threshold(y_test, y_probs, thresh)
        threshold_results.append(result)
        
        print(f"\nThreshold: {thresh:.1f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1:        {result['f1']:.4f}")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Trigger Rate: {result['trigger_rate']*100:.1f}% (would trigger corrections)")
        print(f"  Confusion Matrix: TP={result['true_positives']}, FP={result['false_positives']}, "
              f"TN={result['true_negatives']}, FN={result['false_negatives']}")
    
    # Compare with recommended threshold
    print(f"\n{'─'*80}")
    print("COMPARISON WITH RECOMMENDED THRESHOLD")
    print(f"{'─'*80}")
    rec_result = evaluate_at_threshold(y_test, y_probs, recommended_threshold)
    print(f"Recommended Threshold: {recommended_threshold:.4f}")
    print(f"  Precision: {rec_result['precision']:.4f}")
    print(f"  Recall:    {rec_result['recall']:.4f}")
    print(f"  F1:        {rec_result['f1']:.4f}")
    print(f"  Accuracy:  {rec_result['accuracy']:.4f}")
    print(f"  Trigger Rate: {rec_result['trigger_rate']*100:.1f}%")
    
    # Find best threshold from tested values
    best_f1_idx = np.argmax([r['f1'] for r in threshold_results])
    best_thresh = THRESHOLDS_TO_TEST[best_f1_idx]
    best_result = threshold_results[best_f1_idx]
    
    print(f"\n{'─'*80}")
    print("BEST THRESHOLD (from tested values)")
    print(f"{'─'*80}")
    print(f"Best F1 Threshold: {best_thresh:.1f} (F1={best_result['f1']:.4f})")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}")
    print(f"  Trigger Rate: {best_result['trigger_rate']*100:.1f}%")
    
    # Compile results
    analysis_results = {
        "probe_name": probe_dir.name,
        "model_id": model_id,
        "feature_method": feature_method,
        "classifier": classifier_type,
        "current_recommended_threshold": float(recommended_threshold),
        "test_set_size": int(len(y_test)),
        "prediction_stats": {
            "min": float(np.min(y_probs)),
            "median": float(np.median(y_probs)),
            "max": float(np.max(y_probs)),
            "mean": float(np.mean(y_probs)),
            "std": float(np.std(y_probs)),
        },
        "threshold_results": threshold_results,
        "recommended_threshold_result": rec_result,
        "best_threshold": {
            "threshold": best_thresh,
            "f1": best_result['f1'],
            "precision": best_result['precision'],
            "recall": best_result['recall'],
            "accuracy": best_result['accuracy'],
            "trigger_rate": best_result['trigger_rate'],
        },
    }
    
    return analysis_results, y_test, y_probs

# ============================================================
# Visualization
# ============================================================

def plot_threshold_analysis(analysis_results, y_test, y_probs, output_dir: Path):
    """Create visualization plots for threshold analysis."""
    if not HAS_PLOTTING:
        return
    
    probe_name = analysis_results["probe_name"]
    threshold_results = analysis_results["threshold_results"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Threshold Analysis: {probe_name}", fontsize=14, fontweight='bold')
    
    thresholds = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1_scores = [r['f1'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    trigger_rates = [r['trigger_rate'] * 100 for r in threshold_results]
    
    # 1. Precision, Recall, F1, Accuracy vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(thresholds, precisions, label='Precision', linewidth=2, marker='o', markersize=8)
    ax1.plot(thresholds, recalls, label='Recall', linewidth=2, marker='s', markersize=8)
    ax1.plot(thresholds, f1_scores, label='F1', linewidth=2, marker='^', markersize=8)
    ax1.plot(thresholds, accuracies, label='Accuracy', linewidth=2, marker='d', markersize=8)
    ax1.axvline(analysis_results['current_recommended_threshold'], 
                color='red', linestyle='--', label='Recommended', linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Metrics vs Threshold', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thresholds)
    
    # 2. Trigger rate
    ax2 = axes[0, 1]
    ax2.plot(thresholds, trigger_rates, linewidth=2, color='purple', marker='o', markersize=8)
    ax2.axvline(analysis_results['current_recommended_threshold'], 
                color='red', linestyle='--', label='Recommended', linewidth=2)
    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('Trigger Rate (%)', fontsize=11)
    ax2.set_title('Correction Trigger Rate vs Threshold', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(thresholds)
    
    # 3. Confusion matrix heatmap for best threshold
    ax3 = axes[1, 0]
    best_result = analysis_results['best_threshold']
    best_thresh = best_result['threshold']
    best_idx = thresholds.index(best_thresh)
    best_cm = [
        [threshold_results[best_idx]['true_negatives'], threshold_results[best_idx]['false_positives']],
        [threshold_results[best_idx]['false_negatives'], threshold_results[best_idx]['true_positives']]
    ]
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Low Entropy', 'High Entropy'],
                yticklabels=['Low Entropy', 'High Entropy'])
    ax3.set_title(f'Confusion Matrix (Threshold={best_thresh:.1f})', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=10)
    ax3.set_xlabel('Predicted Label', fontsize=10)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Model: {analysis_results['model_id']}
    Feature Method: {analysis_results['feature_method']}
    
    Current Recommended: {analysis_results['current_recommended_threshold']:.4f}
    
    Best Threshold (F1): {best_result['threshold']:.1f}
    - F1:        {best_result['f1']:.4f}
    - Precision: {best_result['precision']:.4f}
    - Recall:    {best_result['recall']:.4f}
    - Accuracy:  {best_result['accuracy']:.4f}
    - Trigger:   {best_result['trigger_rate']*100:.1f}%
    
    Prediction Range:
    - Min:    {analysis_results['prediction_stats']['min']:.4f}
    - Median: {analysis_results['prediction_stats']['median']:.4f}
    - Max:    {analysis_results['prediction_stats']['max']:.4f}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    output_file = output_dir / f"{probe_name}_threshold_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_file}")
    plt.close()

# ============================================================
# Main Analysis
# ============================================================

def main():
    """Run threshold analysis for all probes."""
    print("\n" + "="*80)
    print("PROBE THRESHOLD TUNING ANALYSIS")
    print("="*80 + "\n")
    print(f"Testing thresholds: {THRESHOLDS_TO_TEST}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Find all probes
    probes = find_all_probes(SAVED_PROBES_DIR)
    if len(probes) == 0:
        print(f"❌ No probes found in {SAVED_PROBES_DIR}")
        print("   Make sure you've run train_probes.py first.")
        return
    
    print(f"Found {len(probes)} probe(s) to analyze\n")
    
    # Find dataset files
    datasets = find_dataset_files(SCRIPT_DIR)
    if len(datasets) == 0:
        print(f"❌ No dataset files found in {SCRIPT_DIR}")
        print("   Make sure you've run train_probes.py to generate datasets.")
        return
    
    print(f"Found {len(datasets)} dataset file(s)\n")
    
    # Analyze each probe
    all_results = []
    
    for probe_dir in probes:
        try:
            # Load metadata to get model info
            metadata_path = probe_dir / "probe_metadata.json"
            if not metadata_path.exists():
                print(f"⚠️  Skipping {probe_dir.name}: No metadata")
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_id = metadata.get("model_id", "")
            model_name_safe = model_id.replace("/", "_")
            
            # Find matching dataset
            if model_name_safe not in datasets:
                print(f"⚠️  Skipping {probe_dir.name}: No matching dataset for {model_id}")
                continue
            
            dataset_type, dataset_path = datasets[model_name_safe]
            print(f"Loading dataset: {dataset_path.name} (format: {dataset_type})")
            
            # Load dataset - we only need task_id and semantic_entropy (not features)
            # Features will be regenerated for test set only
            if dataset_type == "pkl":
                # Pickle files preserve full arrays
                df = pd.read_pickle(dataset_path)
                print(f"✅ Loaded dataset with {len(df)} examples (from pickle)")
            else:
                # CSV files - we only need task_id and semantic_entropy
                # Features will be regenerated for test set tasks only
                print(f"Loading from CSV (only task_id and semantic_entropy needed)...")
                df = pd.read_csv(dataset_path)
                
                # Only keep the columns we need (skip feature columns)
                required_cols = ["task_id", "semantic_entropy"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns in CSV: {missing_cols}")
                
                df = df[required_cols].copy()
                print(f"✅ Loaded dataset metadata with {len(df)} examples (from CSV)")
                print(f"   Features will be regenerated for test set tasks only (using DatasetSplit)")
            
            # Get HF token if needed
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print(f"\n⚠️  HF_TOKEN not set in environment.")
                print(f"   Feature regeneration requires HuggingFace token.")
                print(f"   Set it with: export HF_TOKEN=your_token")
                print(f"   Or login will be prompted when needed")
            
            # Analyze probe (will regenerate test set features)
            split_dir = SCRIPT_DIR / "DatasetSplit"
            result = analyze_probe_thresholds(
                probe_dir, df, split_dir=split_dir, 
                hf_token=hf_token, regenerate_features=True
            )
            
            if result is not None:
                analysis_results, y_test, y_probs = result
                all_results.append(analysis_results)
                
                # Save results
                output_file = OUTPUT_DIR / f"{probe_dir.name}_threshold_analysis.json"
                with open(output_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                print(f"\n✅ Saved analysis to: {output_file}")
                
                # Create plots
                if HAS_PLOTTING:
                    plot_threshold_analysis(analysis_results, y_test, y_probs, OUTPUT_DIR)
            
        except Exception as e:
            print(f"❌ Error analyzing {probe_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary across all probes
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL PROBES")
        print(f"{'='*80}\n")
        
        summary_data = []
        for result in all_results:
            for thresh_result in result["threshold_results"]:
                summary_data.append({
                    "probe": result["probe_name"],
                    "model": result["model_id"],
                    "feature_method": result["feature_method"],
                    "threshold": thresh_result["threshold"],
                    "precision": thresh_result["precision"],
                    "recall": thresh_result["recall"],
                    "f1": thresh_result["f1"],
                    "accuracy": thresh_result["accuracy"],
                    "trigger_rate": thresh_result["trigger_rate"],
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create pivot table for easy comparison
        print("F1 Scores by Threshold:")
        print("-" * 80)
        pivot_f1 = summary_df.pivot_table(
            values='f1', 
            index=['model', 'feature_method'], 
            columns='threshold',
            aggfunc='first'
        )
        print(pivot_f1.to_string())
        
        print("\n\nTrigger Rates (%) by Threshold:")
        print("-" * 80)
        pivot_trigger = summary_df.pivot_table(
            values='trigger_rate', 
            index=['model', 'feature_method'], 
            columns='threshold',
            aggfunc='first'
        ) * 100
        print(pivot_trigger.to_string())
        
        # Save summary
        summary_file = OUTPUT_DIR / "threshold_analysis_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ Saved summary to: {summary_file}")
        
        # Save pivot tables
        pivot_f1_file = OUTPUT_DIR / "threshold_f1_pivot.csv"
        pivot_f1.to_csv(pivot_f1_file)
        print(f"✅ Saved F1 pivot table to: {pivot_f1_file}")
        
        pivot_trigger_file = OUTPUT_DIR / "threshold_trigger_pivot.csv"
        pivot_trigger.to_csv(pivot_trigger_file)
        print(f"✅ Saved trigger rate pivot table to: {pivot_trigger_file}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

