#!/usr/bin/env python3
"""
Main Pipeline Runner

This file orchestrates the full pipeline:

1. Train SEP probe (optional, if not already trained)
2. Run adaptive decoding evaluation
3. Run self-correction on test problems
4. Evaluate and report results
5. Generate comprehensive comparison report
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import torch
import pandas as pd
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

# Import probe training functions
from sep_training_lambda import (
    get_config as get_training_config,
    build_sep_dataset_for_model,
    train_eval_probe_for_model,
    save_model_artifacts,
    relation_report,
    set_seed,
    safe_name,
)

# ============================================================
# Configuration
# ============================================================

def get_pipeline_config():
    return {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "dataset_name": "openai_humaneval",
        "split": "test",
        "limit_tasks": None,  # Set to a number for testing, or None for full 164
        
        # Pipeline steps to run
        "train_probe_if_missing": True,  # Automatically train probe if not found
        "run_adaptive_decoding": True,  # Run adaptive decoding evaluation
        "run_self_correction": True,    # Run self-correction evaluation
        "skip_if_results_exist": True,   # Skip if results files already exist
        
        # SEP probe settings
        "probe_path": "sep_slt_runs/meta-llama_Llama-3.2-3B-Instruct",
        "use_sep_probe": True,
        
        # Probe training settings (if training is needed)
        "probe_training_limit_tasks": None,  # None = full dataset, or set to number for faster training
        
        # Output settings
        "output_dir": "pipeline_results",
        "generate_report": True,
        "save_detailed_results": True,
    }

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_on_humaneval(
    num_problems: Optional[int] = None,
    run_adaptive: bool = True,
    run_correction: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate self-correction and adaptive decoding on HumanEval problems.
    
    Args:
        num_problems: Number of problems to evaluate (None = all)
        run_adaptive: Whether to run adaptive decoding evaluation
        run_correction: Whether to run self-correction evaluation
        cfg: Pipeline configuration
    
    Returns:
        Dictionary with all evaluation results
    """
    if cfg is None:
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
    
    # Load dataset
    print(f"\nLoading HumanEval dataset...")
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    tasks = [ds[i] for i in range(len(ds))]
    
    if num_problems is not None:
        tasks = tasks[:num_problems]
    elif cfg["limit_tasks"] is not None:
        tasks = tasks[:cfg["limit_tasks"]]
    
    print(f"Evaluating on {len(tasks)} HumanEval problems")
    
    # Load model
    print(f"\nLoading model: {cfg['model_id']}")
    tok, model = load_model(cfg["model_id"], hf_token=HF_TOKEN)
    if tok is None:
        print("[ERROR] Could not load model")
        return {"error": "Could not load model"}
    
    # Load or train SEP probe
    sep_probe = None
    if cfg["use_sep_probe"]:
        probe_dir = cfg["probe_path"]
        probe_pkl_path = os.path.join(probe_dir, "probe.pkl")
        
        # Check if probe exists
        if os.path.exists(probe_pkl_path):
            scaler, clf, threshold = load_sep_probe(probe_dir)
            if scaler is not None:
                sep_probe = (scaler, clf, threshold)
                print(f"✅ Using existing SEP probe from {probe_dir}")
                print(f"   Auto-loaded threshold: {threshold:.4f} (will be used for adaptive decoding and self-correction)")
            else:
                print(f"⚠️  Probe file exists but couldn't load, will train new one")
                sep_probe = None
        else:
            print(f"⚠️  SEP probe not found at {probe_dir}")
            sep_probe = None
        
        # Train probe if missing and training is enabled
        if sep_probe is None and cfg.get("train_probe_if_missing", True):
            print("\n" + "="*80)
            print("STEP 0: Training SEP Probe (probe not found)")
            print("="*80)
            print("This will take 60-120 minutes for full dataset...")
            
            # Get training config
            training_cfg = get_training_config()
            training_cfg["model_id"] = cfg["model_id"]
            training_cfg["limit_tasks"] = cfg.get("probe_training_limit_tasks", None)
            set_seed(training_cfg["seed"])
            
            # Build SEP dataset
            print(f"\nBuilding SEP dataset for {cfg['model_id']}...")
            rows = build_sep_dataset_for_model(cfg["model_id"], training_cfg, tasks, hf_token=HF_TOKEN)
            
            if rows is None:
                print("[ERROR] Could not build SEP dataset")
                return {"error": "Could not build SEP dataset"}
            
            df = pd.DataFrame(rows)
            print(f"\n✅ Collected {len(df)} training examples")
            
            # Train probe
            scaler, clf, thr, df_use, recommended_threshold = train_eval_probe_for_model(df, training_cfg)
            
            if scaler is None:
                print("\n[ERROR] Probe training failed")
                return {"error": "Probe training failed"}
            
            # Report semantic entropy vs failure correlation
            relation_report(df)
            
            # Save artifacts (including recommended threshold)
            save_model_artifacts(training_cfg, cfg["model_id"], df_use, scaler, clf, thr, recommended_threshold)
            
            # Load the newly trained probe
            scaler, clf, threshold = load_sep_probe(probe_dir)
            if scaler is not None:
                sep_probe = (scaler, clf, threshold)
                print(f"\n✅ Successfully trained and loaded SEP probe")
                print(f"   Auto-loaded threshold: {threshold:.4f} (will be used for adaptive decoding and self-correction)")
            else:
                print(f"\n[ERROR] Could not load newly trained probe")
                return {"error": "Could not load newly trained probe"}
        
        elif sep_probe is None:
            print(f"⚠️  SEP probe not found and training disabled, evaluations will use fallback methods")
    
    results = {
        "config": cfg,
        "num_tasks": len(tasks),
        "timestamp": datetime.now().isoformat(),
        "baseline_results": None,
        "adaptive_results": None,
        "correction_results": None,
        "comparison": None,
    }
    
    # Baseline evaluation (greedy decoding)
    print("\n" + "="*80)
    print("STEP 1: Baseline Evaluation (Greedy Decoding)")
    print("="*80)
    
    baseline_start = time.time()
    baseline_results = evaluate_baseline(tasks, tok, model, cfg)
    baseline_time = time.time() - baseline_start
    results["baseline_results"] = baseline_results
    results["baseline_time"] = baseline_time
    print(f"\n✅ Baseline Pass@1: {baseline_results['pass_at_1']:.4f}")
    print(f"   Time taken: {baseline_time:.1f} seconds ({baseline_time/60:.1f} minutes)")
    
    # Adaptive decoding evaluation
    if run_adaptive:
        print("\n" + "="*80)
        print("STEP 2: Adaptive Decoding Evaluation")
        print("="*80)
        if sep_probe is not None:
            _, _, probe_threshold = sep_probe
            print(f"   Using auto-loaded threshold: {probe_threshold:.4f}")
        
        adaptive_start = time.time()
        adaptive_cfg = get_adaptive_config()
        adaptive_cfg["limit_tasks"] = len(tasks) if cfg["limit_tasks"] is None else cfg["limit_tasks"]
        adaptive_cfg["probe_path"] = cfg["probe_path"]
        adaptive_cfg["use_sep_probe"] = cfg["use_sep_probe"]
        
        adaptive_results = evaluate_adaptive_decoding(
            tasks, tok, model, adaptive_cfg, sep_probe=sep_probe
        )
        adaptive_time = time.time() - adaptive_start
        results["adaptive_results"] = adaptive_results
        results["adaptive_time"] = adaptive_time
        print(f"\n✅ Adaptive Pass@1: {adaptive_results['adaptive_pass_at_1']:.4f}")
        print(f"   Improvement: {adaptive_results['improvement']:+.4f}")
        print(f"   Time taken: {adaptive_time:.1f} seconds ({adaptive_time/60:.1f} minutes)")
    
    # Self-correction evaluation
    if run_correction:
        print("\n" + "="*80)
        print("STEP 3: Self-Correction Evaluation")
        print("="*80)
        if sep_probe is not None:
            _, _, probe_threshold = sep_probe
            print(f"   Using auto-loaded threshold: {probe_threshold:.4f}")
            print(f"   Config: max_attempts=2, strategy=resample, num_resamples=2 (optimized for speed)")
        
        correction_start = time.time()
        correction_cfg = get_correction_config()
        correction_cfg["limit_tasks"] = len(tasks) if cfg["limit_tasks"] is None else cfg["limit_tasks"]
        correction_cfg["probe_path"] = cfg["probe_path"]
        correction_cfg["use_sep_probe"] = cfg["use_sep_probe"]
        # Threshold will be auto-loaded from probe (set to None in config)
        correction_cfg["uncertainty_threshold"] = None
        
        correction_results = evaluate_self_correction(
            tasks, tok, model, correction_cfg, sep_probe=sep_probe
        )
        correction_time = time.time() - correction_start
        results["correction_results"] = correction_results
        results["correction_time"] = correction_time
        print(f"\n✅ Self-Corrected Pass@1: {correction_results['corrected_pass_at_1']:.4f}")
        print(f"   Improvement: {correction_results['improvement']:+.4f}")
        print(f"   Time taken: {correction_time:.1f} seconds ({correction_time/60:.1f} minutes)")
    
    # Generate comparison
    print("\n" + "="*80)
    print("STEP 4: Generating Comparison Report")
    print("="*80)
    
    comparison = generate_comparison(results)
    results["comparison"] = comparison
    
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
    
    # Latency comparison
    if baseline:
        comparison["latency_comparison"]["baseline"] = baseline["avg_latency"]
    if adaptive:
        comparison["latency_comparison"]["adaptive"] = adaptive["avg_adaptive_latency"]
        if baseline:
            comparison["latency_comparison"]["adaptive_slowdown"] = (
                adaptive["avg_adaptive_latency"] / baseline["avg_latency"]
            )
    if correction:
        comparison["latency_comparison"]["correction"] = correction["avg_corrected_latency"]
        if baseline:
            comparison["latency_comparison"]["correction_slowdown"] = (
                correction["avg_corrected_latency"] / baseline["avg_latency"]
            )
    
    return comparison

# ============================================================
# Reporting Functions
# ============================================================

def print_comprehensive_report(results: Dict[str, Any]):
    """Print a comprehensive report of all results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*80)
    
    baseline = results.get("baseline_results")
    adaptive = results.get("adaptive_results")
    correction = results.get("correction_results")
    comparison = results.get("comparison", {})
    
    print(f"\nEvaluation Date: {results.get('timestamp', 'N/A')}")
    print(f"Number of Tasks: {results.get('num_tasks', 'N/A')}")
    print(f"Model: {results.get('config', {}).get('model_id', 'N/A')}")
    
    # Baseline Results
    if baseline:
        print("\n" + "-"*80)
        print("BASELINE (Greedy Decoding)")
        print("-"*80)
        print(f"Pass@1:           {baseline['pass_at_1']:.4f} ({baseline['pass']}/{baseline['total']})")
        print(f"Avg Latency:      {baseline['avg_latency']:.3f}s")
    
    # Adaptive Decoding Results
    if adaptive:
        print("\n" + "-"*80)
        print("ADAPTIVE DECODING")
        print("-"*80)
        print(f"Pass@1:           {adaptive['adaptive_pass_at_1']:.4f} ({adaptive['adaptive_pass']}/{results['num_tasks']})")
        print(f"Improvement:      {adaptive['improvement']:+.4f} ({adaptive['improvement']*100:+.2f}%)")
        print(f"Avg Latency:      {adaptive['avg_adaptive_latency']:.3f}s")
        print(f"Adaptive Ratio:  {adaptive['avg_adaptive_ratio']:.2%} (fraction of steps using adaptive)")
        print(f"Tasks Improved:   {adaptive['num_improved']}")
        print(f"Tasks Degraded:   {adaptive['num_degraded']}")
        if baseline:
            slowdown = adaptive['avg_adaptive_latency'] / baseline['avg_latency']
            print(f"Latency Slowdown: {slowdown:.2f}x")
    
    # Self-Correction Results
    if correction:
        print("\n" + "-"*80)
        print("SELF-CORRECTION")
        print("-"*80)
        print(f"Pass@1:                {correction['corrected_pass_at_1']:.4f} ({correction['corrected_pass']}/{results['num_tasks']})")
        print(f"Improvement:           {correction['improvement']:+.4f} ({correction['improvement']*100:+.2f}%)")
        print(f"Avg Latency:           {correction['avg_corrected_latency']:.3f}s")
        print(f"Uncertainty Reduction: {correction['avg_uncertainty_reduction']:.4f}")
        print(f"Avg Corrections:       {correction['avg_num_corrections']:.2f} per task")
        print(f"Tasks Improved:        {correction['num_improved']}")
        print(f"Tasks Degraded:       {correction['num_degraded']}")
        if baseline:
            slowdown = correction['avg_corrected_latency'] / baseline['avg_latency']
            print(f"Latency Slowdown:      {slowdown:.2f}x")
    
    # Comparison Summary
    if comparison:
        print("\n" + "-"*80)
        print("COMPARISON SUMMARY")
        print("-"*80)
        
        if comparison.get("improvements"):
            print("\nImprovements over Baseline:")
            for method, improvement in comparison["improvements"].items():
                method_name = method.replace("_vs_baseline", "").replace("_vs_adaptive", " vs Adaptive")
                print(f"  {method_name:30s}: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        print("\nMethod Ranking (by Pass@1):")
        methods = comparison.get("method_comparison", [])
        methods_sorted = sorted(methods, key=lambda x: x["pass_at_1"], reverse=True)
        for i, method in enumerate(methods_sorted, 1):
            print(f"  {i}. {method['method']:25s}: {method['pass_at_1']:.4f}")
    
    print("\n" + "="*80)

def save_results(results: Dict[str, Any], output_dir: str = "pipeline_results"):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = results.get("timestamp", datetime.now().isoformat()).replace(":", "-")
    
    # Save full results
    results_file = os.path.join(output_dir, f"pipeline_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Full results saved to: {results_file}")
    
    # Save comparison summary
    if results.get("comparison"):
        comparison_file = os.path.join(output_dir, f"comparison_{timestamp}.json")
        with open(comparison_file, "w") as f:
            json.dump(results["comparison"], f, indent=2)
        print(f"✅ Comparison saved to: {comparison_file}")
    
    # Save individual method results
    if results.get("baseline_results"):
        baseline_file = os.path.join(output_dir, f"baseline_{timestamp}.json")
        with open(baseline_file, "w") as f:
            json.dump(results["baseline_results"], f, indent=2)
    
    if results.get("adaptive_results"):
        adaptive_file = os.path.join(output_dir, f"adaptive_{timestamp}.json")
        with open(adaptive_file, "w") as f:
            json.dump(results["adaptive_results"], f, indent=2)
    
    if results.get("correction_results"):
        correction_file = os.path.join(output_dir, f"correction_{timestamp}.json")
        with open(correction_file, "w") as f:
            json.dump(results["correction_results"], f, indent=2)
    
    return results_file

# ============================================================
# Main
# ============================================================

def main():
    """Main pipeline runner."""
    print("="*80)
    print("MAIN PIPELINE RUNNER")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Evaluate baseline (greedy decoding)")
    print("  2. Evaluate adaptive decoding (if enabled)")
    print("  3. Evaluate self-correction (if enabled)")
    print("  4. Generate comprehensive comparison report")
    print("="*80)
    
    cfg = get_pipeline_config()
    
    # Check if results already exist
    if cfg["skip_if_results_exist"]:
        output_dir = cfg.get("output_dir", "pipeline_results")
        if os.path.exists(output_dir):
            existing_files = [f for f in os.listdir(output_dir) if f.startswith("pipeline_results_")]
            if existing_files:
                print(f"\n⚠️  Found existing results in {output_dir}")
                response = input("Skip evaluation and load existing results? (y/n): ").strip().lower()
                if response == "y":
                    # Load and display existing results
                    latest_file = max(
                        [os.path.join(output_dir, f) for f in existing_files],
                        key=os.path.getmtime
                    )
                    with open(latest_file, "r") as f:
                        results = json.load(f)
                    print_comprehensive_report(results)
                    return
    
    # Run evaluation
    start_time = time.time()
    
    results = evaluate_on_humaneval(
        num_problems=cfg.get("limit_tasks"),
        run_adaptive=cfg.get("run_adaptive_decoding", True),
        run_correction=cfg.get("run_self_correction", True),
        cfg=cfg,
    )
    
    total_time = time.time() - start_time
    results["total_evaluation_time"] = total_time
    
    # Add timing breakdown
    timing_breakdown = {
        "baseline": results.get("baseline_time", 0),
        "adaptive": results.get("adaptive_time", 0),
        "correction": results.get("correction_time", 0),
        "total": total_time,
    }
    results["timing_breakdown"] = timing_breakdown
    
    print("\n" + "="*80)
    print("TIMING BREAKDOWN")
    print("="*80)
    print(f"Baseline evaluation:    {timing_breakdown['baseline']:.1f}s ({timing_breakdown['baseline']/60:.1f} min)")
    if timing_breakdown['adaptive'] > 0:
        print(f"Adaptive decoding:      {timing_breakdown['adaptive']:.1f}s ({timing_breakdown['adaptive']/60:.1f} min)")
    if timing_breakdown['correction'] > 0:
        print(f"Self-correction:         {timing_breakdown['correction']:.1f}s ({timing_breakdown['correction']/60:.1f} min)")
    print(f"Total pipeline time:    {timing_breakdown['total']:.1f}s ({timing_breakdown['total']/60:.1f} min)")
    print("="*80)
    
    # Generate and print report
    if cfg.get("generate_report", True):
        print_comprehensive_report(results)
    
    # Save results
    if cfg.get("save_detailed_results", True):
        save_results(results, output_dir=cfg.get("output_dir", "pipeline_results"))
    
    print(f"\n✅ Pipeline completed in {total_time:.1f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()

