#!/usr/bin/env python3
"""
DeepSeek-Only Evaluation Script

This script evaluates only DeepSeek-Coder-1.3B-Instruct with both SLT and TBG features
to analyze performance after fixing code extraction issues.
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
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

from run_pipeline_all_models import (
    load_thresholds_from_csv,
    load_test_split,
    evaluate_baseline,
    generate_comparison,
)

# ============================================================
# Configuration
# ============================================================

def get_config():
    return {
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "model_family": "deepseek",
        "hf_token": os.environ.get("HF_TOKEN"),
        
        "dataset_name": "openai_humaneval",
        "split": "test",
        
        "probe_base_dir": "Dataset+Probes/saved_probes",
        "threshold_csv": "threshold_recommendations.csv",
        "split_dir": "Dataset+Probes/DatasetSplit",
        
        "run_baseline": True,
        "run_adaptive_decoding": True,
        "run_self_correction": True,
        
        "test_timeout_s": 10,
        "max_new_tokens": 256,
        "seed": 42,
    }

# ============================================================
# Main Evaluation
# ============================================================

def evaluate_deepseek_only():
    """Evaluate DeepSeek model with both SLT and TBG features."""
    cfg = get_config()
    
    if not cfg["hf_token"]:
        print("⚠️  HF_TOKEN not found in environment. Attempting to login...")
        try:
            login()
            cfg["hf_token"] = os.environ.get("HF_TOKEN")
        except Exception as e:
            print(f"❌ Failed to login: {e}")
            return None
    
    print("="*80)
    print("DEEPSEEK-ONLY EVALUATION")
    print("="*80)
    print(f"Model: {cfg['model_id']}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load thresholds
    threshold_csv = Path(cfg["threshold_csv"])
    thresholds = load_thresholds_from_csv(threshold_csv)
    
    # Load test split
    test_tasks = load_test_split(cfg["split_dir"])
    print(f"\n✅ Loaded {len(test_tasks)} test tasks")
    
    # Results storage
    all_results = []
    
    # Evaluate both SLT and TBG
    for feature_method in ["SLT", "TBG"]:
        print("\n" + "="*80)
        print(f"EVALUATING: {cfg['model_id']} with {feature_method} features")
        print("="*80)
        
        # Get threshold for this probe
        probe_name = f"deepseek-ai_deepseek-coder-1.3b-instruct_{feature_method}_mlp"
        threshold = thresholds.get(probe_name, 0.5)
        print(f"Using threshold: {threshold:.4f}")
        
        # Load probe
        probe_dir = Path(cfg["probe_base_dir"]) / probe_name
        sep_probe = load_sep_probe(str(probe_dir), threshold_override=threshold, feature_method=feature_method)
        
        if sep_probe[0] is None:
            print(f"⚠️  Probe not found at {probe_dir}, skipping {feature_method}")
            continue
        
        # Load model
        print(f"\nLoading model: {cfg['model_id']}")
        tok, model = load_model(cfg["model_id"], cfg["hf_token"])
        print("✅ Model loaded")
        
        # Prepare config
        eval_cfg = {
            "test_timeout_s": cfg["test_timeout_s"],
            "max_new_tokens": cfg["max_new_tokens"],
        }
        
        results = {
            "model_id": cfg["model_id"],
            "model_family": cfg["model_family"],
            "feature_method": feature_method,
            "threshold": threshold,
            "num_tasks": len(test_tasks),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Baseline evaluation
        if cfg["run_baseline"]:
            print("\n" + "-"*80)
            print("STEP 1: Baseline Evaluation (Greedy Decoding)")
            print("-"*80)
            baseline_start = time.time()
            baseline_results = evaluate_baseline(test_tasks, tok, model, eval_cfg, model_id=cfg["model_id"])
            baseline_time = time.time() - baseline_start
            results["baseline_results"] = baseline_results
            results["baseline_time"] = baseline_time
            print(f"\n✅ Baseline Pass@1: {baseline_results['pass_at_1']:.4f}")
            print(f"   Time taken: {baseline_time:.1f} seconds")
            print(f"   Passed: {baseline_results['pass']}/{baseline_results['total']}")
        
        # Adaptive decoding evaluation
        if cfg["run_adaptive_decoding"]:
            print("\n" + "-"*80)
            print("STEP 2: Adaptive Decoding Evaluation")
            print("-"*80)
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
            print(f"   Time taken: {adaptive_time:.1f} seconds")
        
        # Self-correction evaluation
        if cfg["run_self_correction"]:
            print("\n" + "-"*80)
            print("STEP 3: Self-Correction Evaluation")
            print("-"*80)
            correction_start = time.time()
            correction_cfg = get_correction_config()
            correction_cfg["limit_tasks"] = len(test_tasks)
            correction_cfg["use_sep_probe"] = True
            correction_cfg["uncertainty_threshold"] = threshold
            
            correction_results = evaluate_self_correction(
                test_tasks, tok, model, correction_cfg, sep_probe=sep_probe
            )
            correction_time = time.time() - correction_start
            results["correction_results"] = correction_results
            results["correction_time"] = correction_time
            print(f"\n✅ Self-Corrected Pass@1: {correction_results['corrected_pass_at_1']:.4f}")
            print(f"   Improvement: {correction_results['improvement']:+.4f}")
            print(f"   Time taken: {correction_time:.1f} seconds")
        
        # Generate comparison
        comparison = generate_comparison(results)
        results["comparison"] = comparison
        
        all_results.append(results)
        
        # Cleanup model
        del model, tok
        torch.cuda.empty_cache()
        print("\n✅ Evaluation complete for", feature_method)
    
    # Save results
    output_dir = Path("pipeline_results_all_models")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    results_file = output_dir / f"deepseek_only_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("DEEPSEEK-ONLY EVALUATION COMPLETE")
    print("="*80)
    
    # Print summary
    print("\n📊 SUMMARY:")
    for result in all_results:
        feature_method = result["feature_method"]
        baseline = result.get("baseline_results", {})
        adaptive = result.get("adaptive_results", {})
        correction = result.get("correction_results", {})
        
        print(f"\n{feature_method} Features:")
        if baseline:
            print(f"  Baseline:  {baseline.get('pass_at_1', 0):.4f} (Latency: {baseline.get('avg_latency', 0):.3f}s)")
        if adaptive:
            print(f"  Adaptive:  {adaptive.get('adaptive_pass_at_1', 0):.4f} (Improvement: {adaptive.get('improvement', 0):+.4f}, Latency: {adaptive.get('avg_adaptive_latency', 0):.3f}s)")
        if correction:
            print(f"  Corrected: {correction.get('corrected_pass_at_1', 0):.4f} (Improvement: {correction.get('improvement', 0):+.4f}, Latency: {correction.get('avg_corrected_latency', 0):.3f}s)")
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return all_results

if __name__ == "__main__":
    results = evaluate_deepseek_only()

