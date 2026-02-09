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

from utils import get_probe_path, get_token_entropy_threshold, get_task_fields

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
CLASSIFIERS = ["logreg", "mlp"]

# Paths
SCRIPT_DIR = Path(__file__).parent
PROBES_DIR = SCRIPT_DIR / "saved_probes"  # Probes are now in this folder
THRESHOLD_CSV = (
    SCRIPT_DIR / "threshold_recommendations.csv"
)  # Threshold CSV is now in this folder
DATASET_SPLIT_DIR = SCRIPT_DIR / "DatasetSplit"  # Dataset splits are in this folder

# BigCodeBench artifacts from Dataset+Probes/
DATASET_PROBES_DIR = SCRIPT_DIR.parent / "Dataset+Probes"
PROBES_DIR_BIGCODEBENCH = DATASET_PROBES_DIR / "saved_probes" / "bigcodebench"
THRESHOLD_CSV_BIGCODEBENCH = (
    DATASET_PROBES_DIR / "threshold_analysis_plots" / "bigcodebench" / "threshold_recommendations.csv"
)
DATASET_SPLIT_DIR_BIGCODEBENCH = DATASET_PROBES_DIR / "DatasetSplit" / "bigcodebench"


def get_pipeline_config():
    return {
        # Dataset configuration
        # For HumanEval: dataset_name="openai_humaneval", split="test", prompt_field="prompt"
        # For BigCodeBench: dataset_name="bigcode/bigcodebench", split="v0.1.4", prompt_field="instruct_prompt"
        "dataset_name": "bigcode/bigcodebench",  # Changed to BigCodeBench
        "split": "v0.1.4",  # BigCodeBench version
        "prompt_field": "instruct_prompt",  # Which prompt field to use for BigCodeBench
        "limit_tasks": None,  # Set to a number for testing, or None for full dataset
        # Pipeline steps to run
        "run_adaptive_decoding": True,  # Run adaptive decoding evaluation
        "run_self_correction_with_uncertainty": True,  # Run self-correction with uncertainty evaluation
        "run_self_correction_with_verification": True,  # Run self-correction with verification evaluation
        "skip_if_results_exist": False,  # Don't skip - we want all results
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
    Uses best_threshold field (F1-optimized threshold) instead of current_recommended (median).
    """
    if not csv_path.exists():
        print(f"⚠️  Threshold CSV not found at {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    thresholds = {}

    for _, row in df.iterrows():
        probe_name = row["probe"]
        # Use best_threshold (F1-optimized) instead of current_recommended (median)
        best_thresh = row["best_threshold"]
        thresholds[probe_name] = float(best_thresh)

    print(
        f"✅ Loaded {len(thresholds)} thresholds from CSV (using best_threshold/F1-optimized thresholds)"
    )
    return thresholds


def load_test_task_ids(split_dir: Path) -> List[str]:
    """Load test task IDs from DatasetSplit."""
    test_csv = split_dir / "test_tasks.csv"
    if not test_csv.exists():
        print(f"⚠️  Test tasks CSV not found at {test_csv}")
        return None

    df = pd.read_csv(test_csv)
    return df["task_id"].tolist()


def load_dataset_config_from_split(split_dir: Path) -> Optional[Dict[str, str]]:
    """
    Load dataset configuration from split_summary.json.
    
    This ensures we use the same dataset that was used for training the probes.
    """
    summary_file = split_dir / "split_summary.json"
    if not summary_file.exists():
        print(f"⚠️  Split summary not found at {summary_file}")
        return None
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
    
    dataset_config = {
        "dataset_name": summary.get("dataset_name", "openai_humaneval"),
        "split": summary.get("dataset_split", "test"),
        "prompt_field": summary.get("prompt_field", "prompt"),
    }
    
    print(f"✅ Loaded dataset config from split_summary.json:")
    print(f"   Dataset: {dataset_config['dataset_name']}")
    print(f"   Split: {dataset_config['split']}")
    print(f"   Prompt field: {dataset_config['prompt_field']}")
    
    return dataset_config


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
    classifier: str,
) -> Dict[str, Any]:
    """
    Evaluate self-correction and adaptive decoding for a single model/feature combination.
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_id} with {feature_method} features")
    print("=" * 80)

    # Load model
    print(f"\nLoading model: {model_id}")
    tok, model = load_model(model_id, hf_token=hf_token)
    if tok is None:
        print(f"[ERROR] Could not load model {model_id}")
        return {"error": f"Could not load model {model_id}"}

    # Load probe
    probe_dir = get_probe_path(model_id, feature_method, classifier)
    print(f"\nLoading probe from: {probe_dir}")

    if not probe_dir.exists():
        print(f"⚠️  Probe directory not found: {probe_dir}")
        return {"error": f"Probe not found: {probe_dir}"}

    scaler, clf, probe_threshold, probe_feature_method = load_sep_probe(
        str(probe_dir), threshold_override=threshold, feature_method=feature_method
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
        "classifier": classifier,
        "threshold": threshold,
        "num_tasks": len(test_tasks),
        "timestamp": datetime.now().isoformat(),
        "baseline_results": None,
        "adaptive_results": None,
        "uncertainty_correction_results": None,
        "verification_correction_results": None,
        "comparison": None,
    }

    # Baseline evaluation (greedy decoding)
    print("\n" + "-" * 80)
    print("STEP 1: Baseline Evaluation (Greedy Decoding)")
    print("-" * 80)

    baseline_start = time.time()
    baseline_results = evaluate_baseline(test_tasks, tok, model, cfg, model_id=model_id)
    baseline_time = time.time() - baseline_start
    results["baseline_results"] = baseline_results
    results["baseline_time"] = baseline_time
    print(f"\n✅ Baseline Pass@1: {baseline_results['pass_at_1']:.4f}")
    print(
        f"   Time taken: {baseline_time:.1f} seconds ({baseline_time/60:.1f} minutes)"
    )

    baseline_task_by_id = {
        task_result["task_id"]: task_result
        for task_result in baseline_results.get("task_results", [])
        if "task_id" in task_result
    }

    # Adaptive decoding evaluation
    if cfg.get("run_adaptive_decoding", True):
        print("\n" + "-" * 80)
        print("STEP 2: Adaptive Decoding Evaluation")
        print("-" * 80)
        print(f"   Using threshold: {threshold:.4f}")

        adaptive_start = time.time()
        adaptive_cfg = get_adaptive_config()
        adaptive_cfg["limit_tasks"] = len(test_tasks)
        adaptive_cfg["use_sep_probe"] = True
        # Load model-specific token entropy threshold
        adaptive_cfg["fallback_token_entropy_threshold"] = get_token_entropy_threshold(model_id)

        adaptive_results = evaluate_adaptive_decoding(
            test_tasks,
            tok,
            model,
            adaptive_cfg,
            sep_probe=sep_probe,
            baseline_results=baseline_results,
            baseline_task_by_id=baseline_task_by_id,
        )
        adaptive_time = time.time() - adaptive_start
        results["adaptive_results"] = adaptive_results
        results["adaptive_time"] = adaptive_time
        print(f"\n✅ Adaptive Pass@1: {adaptive_results['adaptive_pass_at_1']:.4f}")
        print(f"   Improvement: {adaptive_results['improvement']:+.4f}")
        print(
            f"   Time taken: {adaptive_time:.1f} seconds ({adaptive_time/60:.1f} minutes)"
        )

    # Self-correction with uncertainty evaluation
    if cfg.get("run_self_correction_with_uncertainty", True):
        print("\n" + "-" * 80)
        print("STEP 3: Self-Correction Evaluation")
        print("-" * 80)
        print(f"   Using threshold: {threshold:.4f}")

        correction_start = time.time()
        correction_cfg = get_correction_config()
        correction_cfg["limit_tasks"] = len(test_tasks)
        correction_cfg["use_sep_probe"] = True
        correction_cfg["uncertainty_threshold"] = threshold  # Use threshold from CSV
        correction_cfg["correction_method"] = "uncertainty"

        correction_results = evaluate_self_correction(
            test_tasks,
            tok,
            model,
            correction_cfg,
            sep_probe=sep_probe,
            baseline_results=baseline_results,
            baseline_task_by_id=baseline_task_by_id,
        )
        correction_time = time.time() - correction_start
        results["uncertainty_correction_results"] = correction_results
        results["uncertainty_correction_time"] = correction_time
        print(
            f"\n✅ Uncertainty-based Self-Corrected Pass@1: {correction_results['uncertainty_corrected_pass_at_1']:.4f}"
        )
        print(f"   Improvement: {correction_results['uncertainty_improvement']:+.4f}")
        print(
            f"   Time taken: {correction_time:.1f} seconds ({correction_time/60:.1f} minutes)"
        )

    # Self-correction with verification evaluation
    if cfg.get("run_self_correction_with_verification", True):
        print("\n" + "-" * 80)
        print("STEP 4: Self-Correction with Verification Evaluation")
        print("-" * 80)
        print(f"   Using threshold: {threshold:.4f}")

        correction_start = time.time()
        correction_cfg = get_correction_config()
        correction_cfg["limit_tasks"] = len(test_tasks)
        correction_cfg["use_sep_probe"] = True
        correction_cfg["uncertainty_threshold"] = threshold  # Use threshold from CSV
        correction_cfg["correction_method"] = "verification"

        correction_results = evaluate_self_correction(
            test_tasks,
            tok,
            model,
            correction_cfg,
            sep_probe=sep_probe,
            baseline_results=baseline_results,
            baseline_task_by_id=baseline_task_by_id,
        )
        correction_time = time.time() - correction_start
        results["verification_correction_results"] = correction_results
        results["verification_correction_time"] = correction_time
        print(
            f"\n✅ Verification-based Self-Corrected Pass@1: {correction_results['verification_corrected_pass_at_1']:.4f}"
        )
        print(f"   Improvement: {correction_results['verification_improvement']:+.4f}")
        print(
            f"   Time taken: {correction_time:.1f} seconds ({correction_time/60:.1f} minutes)"
        )

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
    model_id: str = None,  # Add model_id for debugging
) -> Dict[str, Any]:
    """Evaluate baseline greedy decoding."""
    results = {
        "pass": 0,
        "total": len(tasks),
        "latencies": [],
        "task_results": [],
    }

    # Check if this is DeepSeek for debugging
    is_deepseek = model_id and "deepseek" in model_id.lower()

    for task in tqdm(tasks, desc="Baseline evaluation", unit="task"):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_src = task["test"]
        entry_point = task["entry_point"]
        code_prompt = task.get("code_prompt", "")

        user_prompt = prompt + "\n\n# Your code below:\n"
        base_text, _, base_latency = greedy_decode(
            tok, model, user_prompt, max_new_tokens=512
        )
        base_code = extract_code(base_text)

        # DEBUG: For DeepSeek, log first few generations to diagnose issues
        if is_deepseek and len(results["task_results"]) < 3:
            print(f"\n  [DEBUG DeepSeek Task {task_id}]")
            print(f"    Generated text (first 200 chars): {base_text[:200]}...")
            print(f"    Extracted code (first 200 chars): {base_code[:200]}...")
            print(f"    Code length: {len(base_code)}")

        base_correct = evaluate_completion(
            prompt,
            test_src,
            entry_point,
            base_code,
            timeout_s=cfg.get("test_timeout_s", 10),
            code_prompt=code_prompt,
        )

        if base_correct:
            results["pass"] += 1
        results["latencies"].append(base_latency)
        results["task_results"].append(
            {
                "task_id": task_id,
                "correct": base_correct,
                "latency": base_latency,
                "code_length": len(base_code),  # Add code length for debugging
            }
        )

    results["pass_at_1"] = results["pass"] / results["total"]
    results["avg_latency"] = float(
        sum(results["latencies"]) / len(results["latencies"])
    )

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
    uncertainty_correction = results.get("uncertainty_correction_results")
    verification_correction = results.get("verification_correction_results")

    if baseline:
        comparison["summary"]["baseline"] = {
            "pass_at_1": baseline["pass_at_1"],
            "avg_latency": baseline["avg_latency"],
        }
        comparison["method_comparison"].append(
            {
                "method": "Baseline (Greedy)",
                "pass_at_1": baseline["pass_at_1"],
                "avg_latency": baseline["avg_latency"],
                "improvement": 0.0,
            }
        )

    if adaptive:
        comparison["summary"]["adaptive"] = {
            "pass_at_1": adaptive["adaptive_pass_at_1"],
            "avg_latency": adaptive["avg_adaptive_latency"],
            "improvement": adaptive["improvement"],
        }
        comparison["method_comparison"].append(
            {
                "method": "Adaptive Decoding",
                "pass_at_1": adaptive["adaptive_pass_at_1"],
                "avg_latency": adaptive["avg_adaptive_latency"],
                "improvement": adaptive["improvement"],
                "tasks_improved": adaptive["num_improved"],
                "tasks_degraded": adaptive["num_degraded"],
            }
        )
        if baseline:
            comparison["improvements"]["adaptive_vs_baseline"] = adaptive["improvement"]

    if uncertainty_correction:
        comparison["summary"]["uncertainty_self_correction"] = {
            "pass_at_1": uncertainty_correction["uncertainty_corrected_pass_at_1"],
            "avg_latency": uncertainty_correction["uncertainty_avg_corrected_latency"],
            "improvement": uncertainty_correction["uncertainty_improvement"],
            "avg_uncertainty_reduction": uncertainty_correction[
                "uncertainty_avg_uncertainty_reduction"
            ],
            "avg_corrections": uncertainty_correction["uncertainty_avg_num_corrections"],
        }
        comparison["method_comparison"].append(
            {
                "method": "Uncertainty-based Self-Correction",
                "pass_at_1": uncertainty_correction["uncertainty_corrected_pass_at_1"],
                "avg_latency": uncertainty_correction["uncertainty_avg_corrected_latency"],
                "improvement": uncertainty_correction["uncertainty_improvement"],
                "tasks_improved": uncertainty_correction["uncertainty_num_improved"],
                "tasks_degraded": uncertainty_correction["uncertainty_num_degraded"],
                "avg_uncertainty_reduction": uncertainty_correction[
                    "uncertainty_avg_uncertainty_reduction"
                ],
                "avg_corrections": uncertainty_correction["uncertainty_avg_num_corrections"],
            }
        )
        if baseline:
            comparison["improvements"]["uncertainty_correction_vs_baseline"] = (
                uncertainty_correction["uncertainty_improvement"]
            )
        if adaptive:
            uncertainty_correction_vs_adaptive = (
                uncertainty_correction["uncertainty_corrected_pass_at_1"]
                - adaptive["adaptive_pass_at_1"]
            )
            comparison["improvements"][
                "uncertainty_correction_vs_adaptive"
            ] = uncertainty_correction_vs_adaptive

    if verification_correction:
        comparison["summary"]["verification_self_correction"] = {
            "pass_at_1": verification_correction["verification_corrected_pass_at_1"],
            "avg_latency": verification_correction["verification_avg_corrected_latency"],
            "improvement": verification_correction["verification_improvement"],
            "avg_corrections": verification_correction["verification_avg_num_corrections"],
        }
        comparison["method_comparison"].append(
            {
                "method": "Verification-based Self-Correction",
                "pass_at_1": verification_correction[
                    "verification_corrected_pass_at_1"
                ],
                "avg_latency": verification_correction["verification_avg_corrected_latency"],
                "improvement": verification_correction["verification_improvement"],
                "tasks_improved": verification_correction["verification_num_improved"],
                "tasks_degraded": verification_correction["verification_num_degraded"],
                "avg_corrections": verification_correction["verification_avg_num_corrections"],
            }
        )
        if baseline:
            comparison["improvements"]["verification_correction_vs_baseline"] = (
                verification_correction["verification_improvement"]
            )
        if adaptive:
            verification_correction_vs_adaptive = (
                verification_correction["verification_corrected_pass_at_1"]
                - adaptive["adaptive_pass_at_1"]
            )
            comparison["improvements"][
                "verification_correction_vs_adaptive"
            ] = verification_correction_vs_adaptive
        if uncertainty_correction:
            verification_correction_vs_uncertainty_correction = (
                verification_correction["verification_corrected_pass_at_1"]
                - uncertainty_correction["uncertainty_corrected_pass_at_1"]
            )
            comparison["improvements"][
                "verification_correction_vs_uncertainty_correction"
            ] = verification_correction_vs_uncertainty_correction

    return comparison


# ============================================================
# Main Pipeline
# ============================================================


def main():
    """Main pipeline runner for all models."""
    print("=" * 80)
    print("FULL PIPELINE RUNNER - ALL MODELS")
    print("=" * 80)
    print("\nThis pipeline will:")
    print("  1. Load probes from saved_probes/")
    print("  2. Load thresholds from threshold_recommendations.csv")
    print("  3. Use dataset splits from DatasetSplit/")
    print("  4. Load dataset (BigCodeBench or HumanEval) based on split_summary.json")
    print("  5. Evaluate baseline, adaptive decoding, and self-correction")
    print("  6. Generate comprehensive comparison report")
    print("=" * 80)

    cfg = get_pipeline_config()

    # Get HF token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass(
            "Paste your Hugging Face token (with Llama access): "
        ).strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")

    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")

    # Load thresholds
    print(f"\nLoading thresholds from: {THRESHOLD_CSV_BIGCODEBENCH}")
    thresholds_dict = load_thresholds_from_csv(THRESHOLD_CSV_BIGCODEBENCH)

    # Load dataset splits
    print(f"\nLoading dataset splits from: {DATASET_SPLIT_DIR_BIGCODEBENCH}")
    test_task_ids = load_test_task_ids(DATASET_SPLIT_DIR_BIGCODEBENCH)

    # Try to load dataset config from split_summary.json (ensures consistency with probe training)
    print(f"\nLoading dataset configuration...")
    split_dataset_config = load_dataset_config_from_split(DATASET_SPLIT_DIR_BIGCODEBENCH)
    
    # Use split_summary config if available, otherwise fall back to pipeline config
    if split_dataset_config:
        dataset_name = split_dataset_config["dataset_name"]
        dataset_split = split_dataset_config["split"]
        prompt_field = split_dataset_config["prompt_field"]
        print(f"   Using dataset config from split_summary.json")
    else:
        dataset_name = cfg["dataset_name"]
        dataset_split = cfg["split"]
        prompt_field = cfg.get("prompt_field", "prompt")
        print(f"   Using dataset config from pipeline config")
    
    print(f"\nLoading dataset: {dataset_name} (split: {dataset_split})")
    ds = load_dataset(dataset_name, split=dataset_split)
    
    # Apply field mapping to standardize task fields
    # This ensures consistent access regardless of HumanEval vs BigCodeBench
    all_tasks = []
    for i in range(len(ds)):
        raw_task = ds[i]
        # Map fields to standardized format (task_id, prompt, test, entry_point, etc.)
        mapped_task = get_task_fields(raw_task, dataset_name, prompt_field)
        all_tasks.append(mapped_task)
    
    print(f"✅ Loaded {len(all_tasks)} tasks from {dataset_name}")

    # Filter to test set if splits are available
    if test_task_ids is not None:
        test_tasks = filter_tasks_by_split(all_tasks, test_task_ids)
    else:
        print("⚠️  Using all tasks (no split filtering)")
        test_tasks = all_tasks
        if cfg.get("limit_tasks") is not None:
            test_tasks = test_tasks[: cfg["limit_tasks"]]

    # Store all results
    all_results = []

    # Run pipeline for each model and feature method
    for model_family, size_bucket, model_id in MODELS:
        for feature_method in FEATURE_METHODS:
            for classifier in CLASSIFIERS:  
                # Get probe name for threshold lookup
                model_name_safe = model_id.replace("/", "_")
                probe_name = f"{model_name_safe}_{feature_method}_{classifier}"

                # Get threshold
                if probe_name not in thresholds_dict:
                    print(f"\n⚠️  No threshold found for {probe_name}, skipping...")
                    continue

                threshold = thresholds_dict[probe_name]

                # we can't use TBG feature method for self-correction since the features are extracted from the prompt only and hence would be static and the uncertainty estimation would not change from the initial solution.
                if feature_method == "TBG":
                    cfg["run_self_correction_with_uncertainty"] = False
                    cfg["run_self_correction_with_verification"] = False
                else:
                    cfg["run_self_correction_with_uncertainty"] = True
                    cfg["run_self_correction_with_verification"] = True

                # Run evaluation
                try:
                    result = evaluate_on_humaneval_for_model(
                        model_id=model_id,
                        model_family=model_family,
                        feature_method=feature_method,
                        threshold=threshold,
                        test_tasks=test_tasks[:10],
                        cfg=cfg,
                        hf_token=HF_TOKEN,
                        classifier=classifier,
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
    print("\n" + "=" * 80)
    print("SUMMARY - ALL MODELS")
    print("=" * 80)
    print("\nMethods:")
    print("  - Baseline: Greedy decoding (temp=0.0)")
    print("  - Adaptive Decoding: ADADEC-style (TBG probe + lookahead scoring)")
    print("  - Self-Correction: MTE-style iterative resampling (temp=0.0 then 0.3)")
    print("=" * 80)

    for result in all_results:
        model_id = result["model_id"]
        feature_method = result["feature_method"]
        baseline = result.get("baseline_results", {})
        adaptive = result.get("adaptive_results", {})
        uncertainty_correction = result.get("uncertainty_correction_results", {})
        verification_correction = result.get("verification_correction_results", {})

        print(f"\n{model_id} ({feature_method}):")
        if baseline:
            print(f"  Baseline (Greedy):")
            print(f"    Pass@1:  {baseline.get('pass_at_1', 0):.4f}")
            print(f"    Latency: {baseline.get('avg_latency', 0):.3f}s")

        if adaptive:
            print(f"  Adaptive Decoding (ADADEC):")
            print(f"    Pass@1:  {adaptive.get('adaptive_pass_at_1', 0):.4f}")
            print(
                f"    Improvement: {adaptive.get('improvement', 0):+.4f} ({adaptive.get('improvement', 0)*100:+.2f}%)"
            )
            print(f"    Latency: {adaptive.get('avg_adaptive_latency', 0):.3f}s")
            print(
                f"    Adaptive Ratio: {adaptive.get('avg_adaptive_ratio', 0)*100:.1f}%"
            )
            print(f"    Tasks Improved: {adaptive.get('num_improved', 0)}")

        if uncertainty_correction:
            print(f"  Self-Correction (MTE-style Resampling):")
            print(f"    Pass@1:  {uncertainty_correction.get('uncertainty_corrected_pass_at_1', 0):.4f}")
            print(
                f"    Improvement: {uncertainty_correction.get('uncertainty_improvement', 0):+.4f} ({uncertainty_correction.get('uncertainty_improvement', 0)*100:+.2f}%)"
            )
            print(f"    Latency: {uncertainty_correction.get('uncertainty_avg_corrected_latency', 0):.3f}s")
            print(
                f"    Avg Regenerations: {uncertainty_correction.get('uncertainty_avg_num_corrections', 0):.2f}"
            )
            print(f"    Tasks Improved: {uncertainty_correction.get('uncertainty_num_improved', 0)}")
        
        if verification_correction:
            print(f"  Verification-based Self-Correction:")
            print(f"    Pass@1:  {verification_correction.get('verification_corrected_pass_at_1', 0):.4f}")
            print(
                f"    Improvement: {verification_correction.get('verification_improvement', 0):+.4f} ({verification_correction.get('verification_improvement', 0)*100:+.2f}%)"
            )
            print(f"    Latency: {verification_correction.get('verification_avg_corrected_latency', 0):.3f}s")
            print(f"    Tasks Improved: {verification_correction.get('verification_num_improved', 0)}")

    print("\n" + "=" * 80)
    print("✅ Pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
