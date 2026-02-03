#!/usr/bin/env python3
"""
Main Pipeline Runner for All Models

This file orchestrates the full pipeline for all models and feature methods:

1. Load probes from Dataset+Probes/saved_probes/
2. Load thresholds from threshold_recommendations.csv
3. Use dataset splits from Dataset+Probes/DatasetSplit
4. Run adaptive decoding evaluation
5. Run self-correction on test problems
6. Evaluate and report results
7. Generate comprehensive comparison report
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass
from tqdm import tqdm

# Import from our modules
from adaptive_decoding_lambda import (
    load_sep_probe,
    load_model,
    greedy_decode,
    adaptive_decode,
    extract_code,
    evaluate_completion,
    get_config as get_adaptive_config,
    evaluate_adaptive_decoding,
)

from self_correction_lambda import (
    correct_code,
    evaluate_self_correction,
    get_config as get_correction_config,
)

# ============================================================
# Configuration
# ============================================================

# Models to evaluate (from MTEmodels notebook)
MODELS = [
    ("llama", "3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("qwen-coder-instruct", "3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
]

# Feature methods to evaluate
FEATURE_METHODS = ["SLT", "TBG"]

# Paths
SCRIPT_DIR = Path(__file__).parent
PROBES_DIR = SCRIPT_DIR / "saved_probes"  # Probes are now in this folder
THRESHOLD_CSV = SCRIPT_DIR / "threshold_recommendations.csv"  # Threshold CSV is now in this folder
DATASET_SPLIT_DIR = SCRIPT_DIR / "DatasetSplit"  # Dataset splits are in this folder

def get_pipeline_config():
    return {
        "dataset_name": "openai_humaneval",
        "split": "test",
        "limit_tasks": None,  # Set to a number for testing, or None for full 164
        
        # Pipeline steps to run
        "run_adaptive_decoding": True,  # Run adaptive decoding evaluation
        "run_self_correction": True,    # Run self-correction evaluation
        "skip_if_results_exist": False,   # Don't skip - we want all results
        
        # Output settings
        "output_dir": "pipeline_results_all_models",
        "generate_report": True,
        "save_detailed_results": True,
    }

# ============================================================
# Helper Functions
# ============================================================

def load_thresholds_from_csv(csv_path: Path) -> Dict[str, float]:
    """Load thresholds from threshold_recommendations.csv.
    Uses current_recommended field (median threshold) instead of best_threshold.
    """
    if not csv_path.exists():
        print(f"⚠️  Threshold CSV not found at {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    thresholds = {}
    
    for _, row in df.iterrows():
        probe_name = row["probe"]
        # Use current_recommended (median threshold) instead of best_threshold
        current_recommended = row["current_recommended"]
        thresholds[probe_name] = float(current_recommended)
    
    print(f"✅ Loaded {len(thresholds)} thresholds from CSV (using current_recommended/median thresholds)")
    return thresholds

def get_probe_path(model_id: str, feature_method: str) -> Path:
    """Get probe path for a given model and feature method."""
    # Convert model_id to probe directory name format
    model_name_safe = model_id.replace("/", "_")
    probe_dir_name = f"{model_name_safe}_{feature_method}_mlp"
    return PROBES_DIR / probe_dir_name

def load_test_task_ids(split_dir: Path) -> List[str]:
    """Load test task IDs from DatasetSplit."""
    test_csv = split_dir / "test_tasks.csv"
    if not test_csv.exists():
        print(f"⚠️  Test tasks CSV not found at {test_csv}")
        return None
    
    df = pd.read_csv(test_csv)
    return df["task_id"].tolist()

def filter_tasks_by_split(tasks: List[Dict], test_task_ids: List[str]) -> List[Dict]:
    """Filter tasks to only include test set tasks."""
    if test_task_ids is None:
        return tasks
    
    task_dict = {task["task_id"]: task for task in tasks}
    filtered = [task_dict[tid] for tid in test_task_ids if tid in task_dict]
    print(f"✅ Filtered to {len(filtered)} test set tasks (from {len(tasks)} total)")
    return filtered

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_on_humaneval_for_model(
    model_id: str,
    model_family: str,
    feature_method: str,
    threshold: float,
    test_tasks: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    hf_token: str,
) -> Dict[str, Any]:
    """
    Evaluate self-correction and adaptive decoding for a single model/feature combination.
    """
    print("\n" + "="*80)
    print(f"EVALUATING: {model_id} with {feature_method} features")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {model_id}")
    tok, model = load_model(model_id, hf_token=hf_token)
    if tok is None:
        print(f"[ERROR] Could not load model {model_id}")
        return {"error": f"Could not load model {model_id}"}
    
    # Load probe
    probe_dir = get_probe_path(model_id, feature_method)
    print(f"\nLoading probe from: {probe_dir}")
    
    if not probe_dir.exists():
        print(f"⚠️  Probe directory not found: {probe_dir}")
        return {"error": f"Probe not found: {probe_dir}"}
    
    scaler, clf, probe_threshold, probe_feature_method = load_sep_probe(
        str(probe_dir), 
        threshold_override=threshold,
        feature_method=feature_method
    )
    
    if scaler is None:
        print(f"⚠️  Could not load probe from {probe_dir}")
        return {"error": f"Could not load probe from {probe_dir}"}
    
    # Use the threshold from CSV (override)
    sep_probe = (scaler, clf, threshold, feature_method)
    print(f"✅ Using threshold from CSV: {threshold:.4f}")
    
    results = {
        "model_id": model_id,
        "model_family": model_family,
        "feature_method": feature_method,
        "threshold": threshold,
        "num_tasks": len(test_tasks),
        "timestamp": datetime.now().isoformat(),
        "baseline_results": None,
        "adaptive_results": None,
        "correction_results": None,
        "comparison": None,
    }
    
    # Baseline evaluation (greedy decoding)
    print("\n" + "-"*80)
    print("STEP 1: Baseline Evaluation (Greedy Decoding)")
    print("-"*80)
    
    baseline_start = time.time()
    baseline_results = evaluate_baseline(test_tasks, tok, model, cfg)
    baseline_time = time.time() - baseline_start
    results["baseline_results"] = baseline_results
    results["baseline_time"] = baseline_time
    print(f"\n✅ Baseline Pass@1: {baseline_results['pass_at_1']:.4f}")
    print(f"   Time taken: {baseline_time:.1f} seconds ({baseline_time/60:.1f} minutes)")
    
    # Adaptive decoding evaluation
    if cfg.get("run_adaptive_decoding", True):
        print("\n" + "-"*80)
        print("STEP 2: Adaptive Decoding Evaluation")
        print("-"*80)
        print(f"   Using threshold: {threshold:.4f}")
        
        adaptive_start = time.time()
        adaptive_cfg = get_adaptive_config()
        adaptive_cfg["limit_tasks"] = len(test_tasks)
        adaptive_cfg["use_sep_probe"] = True
        
        adaptive_results = evaluate_adaptive_decoding(
            test_tasks, tok, model, adaptive_cfg, sep_probe=sep_probe
        )
        adaptive_time = time.time() - adaptive_start
        results["adaptive_results"] = adaptive_results
        results["adaptive_time"] = adaptive_time
        print(f"\n✅ Adaptive Pass@1: {adaptive_results['adaptive_pass_at_1']:.4f}")
        print(f"   Improvement: {adaptive_results['improvement']:+.4f}")
        print(f"   Time taken: {adaptive_time:.1f} seconds ({adaptive_time/60:.1f} minutes)")
    
    # Self-correction evaluation
    if cfg.get("run_self_correction", True):
        print("\n" + "-"*80)
        print("STEP 3: Self-Correction Evaluation")
        print("-"*80)
        print(f"   Using threshold: {threshold:.4f}")
        
        correction_start = time.time()
        correction_cfg = get_correction_config()
        correction_cfg["limit_tasks"] = len(test_tasks)
        correction_cfg["use_sep_probe"] = True
        correction_cfg["uncertainty_threshold"] = threshold  # Use threshold from CSV
        
        correction_results = evaluate_self_correction(
            test_tasks, tok, model, correction_cfg, sep_probe=sep_probe
        )
        correction_time = time.time() - correction_start
        results["correction_results"] = correction_results
        results["correction_time"] = correction_time
        print(f"\n✅ Self-Corrected Pass@1: {correction_results['corrected_pass_at_1']:.4f}")
        print(f"   Improvement: {correction_results['improvement']:+.4f}")
        print(f"   Time taken: {correction_time:.1f} seconds ({correction_time/60:.1f} minutes)")
    
    # Generate comparison
    comparison = generate_comparison(results)
    results["comparison"] = comparison
    
    # Cleanup model
    del model, tok
    torch.cuda.empty_cache()
    
    return results

def evaluate_baseline(
    tasks: List[Dict[str, Any]],
    tok,
    model,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate baseline greedy decoding."""
    results = {
        "pass": 0,
        "total": len(tasks),
        "latencies": [],
        "task_results": [],
    }
    
    for task in tqdm(tasks, desc="Baseline evaluation", unit="task"):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_src = task["test"]
        entry_point = task["entry_point"]
        
        user_prompt = prompt + "\n\n# Your code below:\n"
        base_text, _, base_latency = greedy_decode(
            tok, model, user_prompt, max_new_tokens=256
        )
        base_code = extract_code(base_text)
        base_correct = evaluate_completion(
            prompt, test_src, entry_point, base_code, timeout_s=cfg.get("test_timeout_s", 10)
        )
        
        if base_correct:
            results["pass"] += 1
        results["latencies"].append(base_latency)
        results["task_results"].append({
            "task_id": task_id,
            "correct": base_correct,
            "latency": base_latency,
        })
    
    results["pass_at_1"] = results["pass"] / results["total"]
    results["avg_latency"] = float(sum(results["latencies"]) / len(results["latencies"]))
    
    return results

def generate_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive comparison across all methods."""
    comparison = {
        "summary": {},
        "method_comparison": [],
        "improvements": {},
        "latency_comparison": {},
    }
    
    baseline = results.get("baseline_results")
    adaptive = results.get("adaptive_results")
    correction = results.get("correction_results")
    
    if baseline:
        comparison["summary"]["baseline"] = {
            "pass_at_1": baseline["pass_at_1"],
            "avg_latency": baseline["avg_latency"],
        }
        comparison["method_comparison"].append({
            "method": "Baseline (Greedy)",
            "pass_at_1": baseline["pass_at_1"],
            "avg_latency": baseline["avg_latency"],
            "improvement": 0.0,
        })
    
    if adaptive:
        comparison["summary"]["adaptive"] = {
            "pass_at_1": adaptive["adaptive_pass_at_1"],
            "avg_latency": adaptive["avg_adaptive_latency"],
            "improvement": adaptive["improvement"],
        }
        comparison["method_comparison"].append({
            "method": "Adaptive Decoding",
            "pass_at_1": adaptive["adaptive_pass_at_1"],
            "avg_latency": adaptive["avg_adaptive_latency"],
            "improvement": adaptive["improvement"],
            "tasks_improved": adaptive["num_improved"],
            "tasks_degraded": adaptive["num_degraded"],
        })
        if baseline:
            comparison["improvements"]["adaptive_vs_baseline"] = adaptive["improvement"]
    
    if correction:
        comparison["summary"]["self_correction"] = {
            "pass_at_1": correction["corrected_pass_at_1"],
            "avg_latency": correction["avg_corrected_latency"],
            "improvement": correction["improvement"],
            "avg_uncertainty_reduction": correction["avg_uncertainty_reduction"],
            "avg_corrections": correction["avg_num_corrections"],
        }
        comparison["method_comparison"].append({
            "method": "Self-Correction",
            "pass_at_1": correction["corrected_pass_at_1"],
            "avg_latency": correction["avg_corrected_latency"],
            "improvement": correction["improvement"],
            "tasks_improved": correction["num_improved"],
            "tasks_degraded": correction["num_degraded"],
            "avg_uncertainty_reduction": correction["avg_uncertainty_reduction"],
            "avg_corrections": correction["avg_num_corrections"],
        })
        if baseline:
            comparison["improvements"]["correction_vs_baseline"] = correction["improvement"]
        if adaptive:
            correction_vs_adaptive = correction["corrected_pass_at_1"] - adaptive["adaptive_pass_at_1"]
            comparison["improvements"]["correction_vs_adaptive"] = correction_vs_adaptive
    
    return comparison

# ============================================================
# Main Pipeline
# ============================================================

def main():
    """Main pipeline runner for all models."""
    print("="*80)
    print("FULL PIPELINE RUNNER - ALL MODELS")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Load probes from Dataset+Probes/saved_probes/")
    print("  2. Load thresholds from threshold_recommendations.csv")
    print("  3. Use dataset splits from Dataset+Probes/DatasetSplit")
    print("  4. Evaluate baseline, adaptive decoding, and self-correction")
    print("  5. Generate comprehensive comparison report")
    print("="*80)
    
    cfg = get_pipeline_config()
    
    # Get HF token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with Llama access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    
    # Load thresholds
    print(f"\nLoading thresholds from: {THRESHOLD_CSV}")
    thresholds_dict = load_thresholds_from_csv(THRESHOLD_CSV)
    
    # Load dataset splits
    print(f"\nLoading dataset splits from: {DATASET_SPLIT_DIR}")
    test_task_ids = load_test_task_ids(DATASET_SPLIT_DIR)
    
    # Load full dataset
    print(f"\nLoading HumanEval dataset...")
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    all_tasks = [ds[i] for i in range(len(ds))]
    
    # Filter to test set if splits are available
    if test_task_ids is not None:
        test_tasks = filter_tasks_by_split(all_tasks, test_task_ids)
    else:
        print("⚠️  Using all tasks (no split filtering)")
        test_tasks = all_tasks
        if cfg.get("limit_tasks") is not None:
            test_tasks = test_tasks[:cfg["limit_tasks"]]
    
    # Store all results
    all_results = []
    
    # Run pipeline for each model and feature method
    for model_family, size_bucket, model_id in MODELS:
        for feature_method in FEATURE_METHODS:
            # Get probe name for threshold lookup
            model_name_safe = model_id.replace("/", "_")
            probe_name = f"{model_name_safe}_{feature_method}_mlp"
            
            # Get threshold
            if probe_name not in thresholds_dict:
                print(f"\n⚠️  No threshold found for {probe_name}, skipping...")
                continue
            
            threshold = thresholds_dict[probe_name]
            
            # Run evaluation
            try:
                result = evaluate_on_humaneval_for_model(
                    model_id=model_id,
                    model_family=model_family,
                    feature_method=feature_method,
                    threshold=threshold,
                    test_tasks=test_tasks,
                    cfg=cfg,
                    hf_token=HF_TOKEN,
                )
                
                if "error" not in result:
                    all_results.append(result)
            except Exception as e:
                print(f"\n❌ Error evaluating {model_id} with {feature_method}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save all results
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().isoformat().replace(":", "-")
    
    # Save combined results
    results_file = output_dir / f"all_models_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS")
    print("="*80)
    
    for result in all_results:
        model_id = result["model_id"]
        feature_method = result["feature_method"]
        baseline = result.get("baseline_results", {})
        adaptive = result.get("adaptive_results", {})
        correction = result.get("correction_results", {})
        
        print(f"\n{model_id} ({feature_method}):")
        if baseline:
            print(f"  Baseline:  {baseline.get('pass_at_1', 0):.4f}")
        if adaptive:
            print(f"  Adaptive:  {adaptive.get('adaptive_pass_at_1', 0):.4f} ({adaptive.get('improvement', 0):+.4f})")
        if correction:
            print(f"  Corrected: {correction.get('corrected_pass_at_1', 0):.4f} ({correction.get('improvement', 0):+.4f})")
    
    print("\n" + "="*80)
    print("✅ Pipeline completed!")
    print("="*80)

if __name__ == "__main__":
    main()

