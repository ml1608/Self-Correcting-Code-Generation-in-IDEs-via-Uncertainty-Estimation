#!/usr/bin/env python3
"""
Compute Pearson Correlation Scores for TBG+MLP and SLT+MLP Probes

This script:
1. Loads test set task IDs from DatasetSplit
2. Loads probe dataset CSV files
3. Filters to test set examples
4. Loads probes (TBG+MLP and SLT+MLP) for each model
5. Computes Pearson correlation between probe predictions and semantic entropy
6. Saves results to CSV
"""

import os
import json
import pickle
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from getpass import getpass

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent

# Paths - look in same directory first, then parent directory, can be overridden via environment variables
SAVED_PROBES_DIR = Path(os.environ.get("SAVED_PROBES_DIR", str(SCRIPT_DIR / "saved_probes")))
DATASET_SPLIT_DIR = Path(os.environ.get("DATASET_SPLIT_DIR", str(SCRIPT_DIR / "DatasetSplit")))
# CSV files are in Dataset+Probes folder - try parent/Dataset+Probes or just parent
DATASET_CSV_DIR = Path(os.environ.get("DATASET_CSV_DIR", str(SCRIPT_DIR.parent / "Dataset+Probes")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(SCRIPT_DIR)))

# Models to evaluate
MODELS = [
    ("llama", "3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("qwen-coder-instruct", "3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
]

# Feature methods to evaluate
FEATURE_METHODS = ["SLT", "TBG"]

# Layers used in training (from train_probes.py)
LAYERS = [-3, -2, -1]

# ============================================================
# Helper Functions
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
    )

def load_test_task_ids(split_dir: Path):
    """Load test task IDs from DatasetSplit."""
    test_csv = split_dir / "test_tasks.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test tasks CSV not found at {test_csv}")
    
    df = pd.read_csv(test_csv)
    return set(df["task_id"].tolist())

def load_saved_probe(probe_dir: Path):
    """Load a saved probe from disk."""
    probe_pkl_path = probe_dir / "probe.pkl"
    
    if not probe_pkl_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_pkl_path}")
    
    with open(probe_pkl_path, "rb") as f:
        probe_data = pickle.load(f)
    
    scaler = probe_data["scaler"]
    clf = probe_data["classifier"]
    
    probe_kind = probe_data.get("probe_kind", "classification")
    classifier_name = type(clf).__name__
    return scaler, clf, probe_kind, classifier_name

def get_probe_path(model_id: str, feature_method: str) -> Path:
    """Get probe path for a given model and feature method."""
    model_name_safe = model_id.replace("/", "_")
    probe_dir_name = f"{model_name_safe}_{feature_method}_mlp"
    return SAVED_PROBES_DIR / probe_dir_name

def get_dataset_csv_path(model_id: str) -> Path:
    """Get dataset CSV path for a given model."""
    model_name_safe = model_id.replace("/", "_")
    csv_name = f"probe_dataset_{model_name_safe}.csv"
    # Try in DATASET_CSV_DIR first, then fallback to SCRIPT_DIR
    csv_path = DATASET_CSV_DIR / csv_name
    if not csv_path.exists():
        csv_path = SCRIPT_DIR / csv_name
    return csv_path

def regenerate_test_set_features(model_id: str, test_task_ids: set, feature_method: str, 
                                 layers: list, hf_token: str = None):
    """
    Regenerate features for test set tasks only.
    This is needed when CSV arrays are truncated.
    """
    print(f"  Regenerating features for {len(test_task_ids)} test set tasks...")
    print(f"  Loading model (this may take a moment)...")
    
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

# ============================================================
# Main Computation
# ============================================================

def compute_pearson_for_probe(model_id: str, model_family: str, feature_method: str, 
                              test_task_ids: set, hf_token: str = None):
    """
    Compute Pearson correlation for a specific probe on test set.
    
    Returns:
        dict with model_id, model_family, feature_method, pearson_r, pearson_p, n_samples
    """
    print(f"\n{'─'*80}")
    print(f"Computing Pearson for: {model_id} - {feature_method}+MLP")
    print(f"{'─'*80}")
    
    # Load dataset CSV
    dataset_csv = get_dataset_csv_path(model_id)
    if not dataset_csv.exists():
        print(f"⚠️  Dataset CSV not found: {dataset_csv}")
        return None
    
    print(f"Loading dataset from: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    
    # Filter to test set
    df_test = df[df["task_id"].isin(test_task_ids)].copy()
    print(f"Test set size: {len(df_test)} examples")
    
    if len(df_test) == 0:
        print(f"⚠️  No test examples found for {model_id}")
        return None
    
    # Try to parse features from CSV
    print(f"Attempting to parse {feature_method} features from CSV...")
    feature_arrays = []
    valid_task_ids = []
    need_regeneration = False
    
    for idx, row in df_test.iterrows():
        try:
            arr = parse_array_string(row[feature_method])
            feature_arrays.append(arr)
            valid_task_ids.append(row["task_id"])
        except ValueError as e:
            if "truncated" in str(e).lower() or "..." in str(e):
                need_regeneration = True
                break
            else:
                print(f"⚠️  Failed to parse features for {row['task_id']}: {e}")
                continue
        except Exception as e:
            print(f"⚠️  Failed to parse features for {row['task_id']}: {e}")
            continue
    
    # If arrays are truncated, regenerate features
    if need_regeneration or len(feature_arrays) == 0:
        print(f"⚠️  Features are truncated in CSV or missing. Regenerating features...")
        test_features_dict = regenerate_test_set_features(
            model_id, test_task_ids, feature_method, LAYERS, hf_token
        )
        
        # Build feature arrays and get semantic entropy for regenerated features
        feature_arrays = []
        semantic_entropy_values = []
        valid_task_ids = []
        
        for task_id in test_task_ids:
            if task_id in test_features_dict:
                feature_arrays.append(test_features_dict[task_id])
                valid_task_ids.append(task_id)
                # Get semantic entropy from CSV
                row = df_test[df_test["task_id"] == task_id]
                if len(row) > 0:
                    semantic_entropy_values.append(row.iloc[0]["semantic_entropy"])
                else:
                    print(f"⚠️  No semantic entropy found for {task_id}")
                    continue
        
        if len(feature_arrays) == 0:
            print(f"⚠️  No valid features found for {model_id} - {feature_method}")
            return None
        
        X_test = np.stack(feature_arrays).astype(np.float32)
        semantic_entropy = np.array(semantic_entropy_values)
        df_test_valid = df_test[df_test["task_id"].isin(valid_task_ids)].reset_index(drop=True)
    else:
        # Use features from CSV
        X_test = np.stack(feature_arrays).astype(np.float32)
        df_test_valid = df_test[df_test["task_id"].isin(valid_task_ids)].reset_index(drop=True)
        semantic_entropy = df_test_valid["semantic_entropy"].values
    
    print(f"Valid test examples: {len(X_test)}")
    print(f"Feature dimensions: {X_test.shape[1]}")
    
    # Load probe
    probe_dir = get_probe_path(model_id, feature_method)
    if not probe_dir.exists():
        print(f"⚠️  Probe not found: {probe_dir}")
        return None
    
    print(f"Loading probe from: {probe_dir}")
    scaler, clf, probe_kind, classifier_name = load_saved_probe(probe_dir)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Get probe predictions:
    # - classification probes: P(high entropy)
    # - regression probes: predicted semantic entropy
    print("Computing probe predictions...")
    if hasattr(clf, "predict_proba"):
        uncertainty_score = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        uncertainty_score = clf.predict(X_test_scaled)
    
    # Correlation with semantic entropy (should be positive if probe works)
    print("Computing correlation with semantic entropy...")
    pearson_r_entropy, pearson_p_entropy = pearsonr(uncertainty_score, semantic_entropy)
    spearman_r_entropy, spearman_p_entropy = spearmanr(uncertainty_score, semantic_entropy)
    
    # Correlation with pass@1 (should be negative: high uncertainty -> lower pass rate)
    pass_at_1 = df_test_valid["pass_at_1"].values if "pass_at_1" in df_test_valid.columns else None
    if pass_at_1 is not None:
        pearson_r_pass, pearson_p_pass = pearsonr(uncertainty_score, pass_at_1)
        spearman_r_pass, spearman_p_pass = spearmanr(uncertainty_score, pass_at_1)
    else:
        pearson_r_pass, pearson_p_pass = float("nan"), float("nan")
        spearman_r_pass, spearman_p_pass = float("nan"), float("nan")
    
    print(f"✅ Entropy correlation: pearson={pearson_r_entropy:.4f}, spearman={spearman_r_entropy:.4f}")
    print(f"✅ Pass@1 correlation:  pearson={pearson_r_pass:.4f}, spearman={spearman_r_pass:.4f}")
    print(f"   N samples: {len(uncertainty_score)}")
    
    return {
        "model_id": model_id,
        "model_family": model_family,
        "feature_method": feature_method,
        "probe_type": f"{feature_method}+{classifier_name}",
        "probe_kind": probe_kind,
        "pearson_r_entropy": float(pearson_r_entropy),
        "pearson_p_entropy": float(pearson_p_entropy),
        "spearman_r_entropy": float(spearman_r_entropy),
        "spearman_p_entropy": float(spearman_p_entropy),
        "pearson_r_pass_at_1": float(pearson_r_pass),
        "pearson_p_pass_at_1": float(pearson_p_pass),
        "spearman_r_pass_at_1": float(spearman_r_pass),
        "spearman_p_pass_at_1": float(spearman_p_pass),
        "n_samples": int(len(uncertainty_score)),
    }

def main():
    """Main function to compute Pearson scores for all probes."""
    print("="*80)
    print("PEARSON CORRELATION COMPUTATION FOR PROBES")
    print("="*80)
    
    # Verify required directories exist
    if not SAVED_PROBES_DIR.exists():
        raise FileNotFoundError(
            f"Probes directory not found: {SAVED_PROBES_DIR}\n"
            f"Set SAVED_PROBES_DIR environment variable to point to the saved_probes folder."
        )
    
    if not DATASET_SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Dataset split directory not found: {DATASET_SPLIT_DIR}\n"
            f"Set DATASET_SPLIT_DIR environment variable to point to the DatasetSplit folder."
        )
    
    print(f"Probes directory: {SAVED_PROBES_DIR}")
    print(f"Dataset split directory: {DATASET_SPLIT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get HF token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with model access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Load test task IDs
    print(f"\nLoading test task IDs from: {DATASET_SPLIT_DIR}")
    test_task_ids = load_test_task_ids(DATASET_SPLIT_DIR)
    print(f"✅ Loaded {len(test_task_ids)} test task IDs")
    
    # Compute Pearson for all model/feature combinations
    all_results = []
    
    for model_family, model_size, model_id in MODELS:
        for feature_method in FEATURE_METHODS:
            result = compute_pearson_for_probe(
                model_id, model_family, feature_method, test_task_ids, HF_TOKEN
            )
            if result is not None:
                all_results.append(result)
    
    # Create results DataFrame
    if len(all_results) == 0:
        print("\n❌ No results computed!")
        return
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["model_id", "feature_method"])
    
    # Print summary
    print(f"\n{'='*80}")
    print("PEARSON CORRELATION RESULTS")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))
    
    # Save results
    output_csv = OUTPUT_DIR / "pearson_scores.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    # Also save as JSON
    output_json = OUTPUT_DIR / "pearson_scores.json"
    results_df.to_json(output_json, indent=2, orient="records")
    print(f"✅ Results saved to: {output_json}")

if __name__ == "__main__":
    main()

