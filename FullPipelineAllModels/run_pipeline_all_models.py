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
import random
import numpy as np
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
# Proactive regeneration uses the AdaDec submodule-backed implementation.
from adaptive_decoding_adadec import (
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

from utils import (
    get_probe_path,
    get_token_entropy_threshold,
    get_task_fields,
    get_artifact_paths,
    infer_model_family,
)

# ============================================================
# Defaults (overridable via CLI)
# ============================================================

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
]
DEFAULT_FEATURE_METHODS = ["SLT", "TBG"]
DEFAULT_CLASSIFIERS = ["linreg"]

SCRIPT_DIR = Path(__file__).parent


def get_pipeline_config(args=None):
    """
    Build pipeline config from CLI args.  All values can be overridden on the
    command line; the defaults below match the original BigCodeBench setup.
    """
    return {
        "dataset_name": args.dataset if args and args.dataset else "bigcode/bigcodebench",
        "split": args.split if args and args.split else "v0.1.4",
        "prompt_field": args.prompt_field if args and args.prompt_field else "instruct_prompt",
        "limit_tasks": args.limit_tasks if args else None,
        "run_adaptive_decoding": True,
        "run_self_correction_with_uncertainty": True,
        "run_self_correction_with_verification": True,
        "skip_if_results_exist": False,
        "output_dir": args.output_dir if args and args.output_dir else "pipeline_results_all_models",
        "generate_report": True,
        "save_detailed_results": True,
        "adaptive_trials": args.adaptive_trials if args else 1,
        "bootstrap_samples": args.bootstrap_samples if args else 1000,
        "ci_level": args.ci_level if args else 0.95,
        "adadec_thresholds_json": args.adadec_thresholds_json if args else None,
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


def bootstrap_pass_metrics(
    task_results: List[Dict[str, Any]],
    baseline_key: str,
    adaptive_key: str,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Bootstrap CIs for baseline/adaptive pass@1 and improvement."""
    if not task_results:
        return {}

    pairs = [
        (int(r.get(baseline_key, False)), int(r.get(adaptive_key, False)))
        for r in task_results
    ]
    n = len(pairs)
    base = np.array([p[0] for p in pairs], dtype=np.float64)
    ada = np.array([p[1] for p in pairs], dtype=np.float64)
    improvement = ada - base

    alpha = 1.0 - ci_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    base_boot, ada_boot, imp_boot = [], [], []
    for _ in range(max(1, n_bootstrap)):
        idx = np.random.randint(0, n, size=n)
        base_boot.append(float(np.mean(base[idx])))
        ada_boot.append(float(np.mean(ada[idx])))
        imp_boot.append(float(np.mean(improvement[idx])))

    return {
        "n_bootstrap": int(max(1, n_bootstrap)),
        "ci_level": float(ci_level),
        "baseline_pass_at_1_ci": [float(np.percentile(base_boot, lo_q)), float(np.percentile(base_boot, hi_q))],
        "adaptive_pass_at_1_ci": [float(np.percentile(ada_boot, lo_q)), float(np.percentile(ada_boot, hi_q))],
        "improvement_ci": [float(np.percentile(imp_boot, lo_q)), float(np.percentile(imp_boot, hi_q))],
    }


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
    probes_dir: Path = None,
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
    probe_dir = get_probe_path(model_id, feature_method, classifier, probes_dir=probes_dir)
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
        trial_results = []
        n_trials = max(1, int(cfg.get("adaptive_trials", 1)))

        for trial_idx in range(n_trials):
            random.seed(42 + trial_idx)
            np.random.seed(42 + trial_idx)
            torch.manual_seed(42 + trial_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42 + trial_idx)

            adaptive_cfg = get_adaptive_config()
            adaptive_cfg["limit_tasks"] = len(test_tasks)
            adaptive_cfg["use_sep_probe"] = True
            adaptive_cfg["seed"] = 42 + trial_idx
            # Load model-specific (optionally dataset-aware) AdaDec token threshold
            adaptive_cfg["fallback_token_entropy_threshold"] = get_token_entropy_threshold(
                model_id,
                dataset_name=cfg.get("dataset_name"),
                thresholds_path=cfg.get("adadec_thresholds_json"),
            )

            trial_res = evaluate_adaptive_decoding(
                test_tasks,
                tok,
                model,
                adaptive_cfg,
                sep_probe=sep_probe,
                baseline_results=baseline_results,
                baseline_task_by_id=baseline_task_by_id,
            )
            trial_results.append(trial_res)

        adaptive_results = trial_results[0]
        if n_trials > 1:
            pass_rates = [r["adaptive_pass_at_1"] for r in trial_results]
            improvements = [r["improvement"] for r in trial_results]
            lats = [r["avg_adaptive_latency"] for r in trial_results]
            adaptive_results["multi_trial"] = {
                "n_trials": n_trials,
                "adaptive_pass_at_1_mean": float(np.mean(pass_rates)),
                "adaptive_pass_at_1_std": float(np.std(pass_rates)),
                "improvement_mean": float(np.mean(improvements)),
                "improvement_std": float(np.std(improvements)),
                "avg_adaptive_latency_mean": float(np.mean(lats)),
                "avg_adaptive_latency_std": float(np.std(lats)),
            }

        adaptive_results["error_bars"] = bootstrap_pass_metrics(
            adaptive_results.get("task_results", []),
            baseline_key="baseline_correct",
            adaptive_key="adaptive_correct",
            n_bootstrap=max(1, int(cfg.get("bootstrap_samples", 1000))),
            ci_level=float(cfg.get("ci_level", 0.95)),
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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the full SEP evaluation pipeline on any dataset and model."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL_ID",
        help="HuggingFace model IDs to evaluate. "
             f"Defaults to: {DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="HuggingFace dataset name (e.g. bigcode/bigcodebench or openai_humaneval). "
             "If omitted, read from split_summary.json in the split directory.",
    )
    parser.add_argument(
        "--split", default=None,
        help="Dataset split (e.g. v0.1.4 or test). "
             "If omitted, read from split_summary.json.",
    )
    parser.add_argument(
        "--prompt_field", default=None,
        help="Field name for the prompt (e.g. instruct_prompt or prompt). "
             "If omitted, read from split_summary.json.",
    )
    parser.add_argument(
        "--probes_dir", default=None,
        help="Path to the directory containing saved probe subdirectories. "
             "Defaults to Dataset+Probes/saved_probes/<dataset_safe_name>/.",
    )
    parser.add_argument(
        "--feature_methods", nargs="+", default=DEFAULT_FEATURE_METHODS,
        choices=["SLT", "TBG"],
        help="Feature extraction methods to evaluate.",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS,
        choices=["mlp", "logreg", "linreg"],
        help="Probe classifier types to evaluate.",
    )
    parser.add_argument(
        "--limit_tasks", type=int, default=None,
        help="Limit the number of test tasks (useful for quick runs).",
    )
    parser.add_argument(
        "--output_dir", default="pipeline_results_all_models",
        help="Directory to write result JSON files.",
    )
    parser.add_argument(
        "--threshold_csv", default=None,
        help="Optional override for threshold_recommendations.csv path.",
    )
    parser.add_argument(
        "--split_dir", default=None,
        help="Optional override for DatasetSplit directory path.",
    )
    parser.add_argument(
        "--adaptive_trials", type=int, default=1,
        help="Number of adaptive-decoding evaluation trials to run.",
    )
    parser.add_argument(
        "--bootstrap_samples", type=int, default=1000,
        help="Bootstrap samples for adaptive pass@1/improvement confidence intervals.",
    )
    parser.add_argument(
        "--ci_level", type=float, default=0.95,
        help="Confidence level for bootstrap CIs (e.g. 0.95).",
    )
    parser.add_argument(
        "--adadec_thresholds_json",
        default=None,
        help="Optional JSON path for AdaDec token-level entropy thresholds. "
             "Supports either flat {model_key: value} or dataset-aware "
             "{dataset: {model_key: value}} format.",
    )
    return parser.parse_args()


def main():
    """Main pipeline runner — supports any dataset and any HuggingFace model."""
    args = parse_args()

    print("=" * 80)
    print("FULL PIPELINE RUNNER")
    print("=" * 80)

    cfg = get_pipeline_config(args)

    models = args.models if args.models else DEFAULT_MODELS
    feature_methods = args.feature_methods
    classifiers = args.classifiers

    # Get HF token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token: ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")

    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")

    # Resolve artifact paths from the dataset name (or CLI override)
    # We first try to read from split_summary.json so that the pipeline always
    # uses the same dataset that the probes were trained on.
    artifact_paths = get_artifact_paths(cfg["dataset_name"])
    split_dir = (
        Path(args.split_dir)
        if args.split_dir
        else (
            Path(args.probes_dir).parent.parent / "DatasetSplit" / artifact_paths["split_dir"].name
            if args.probes_dir
            else artifact_paths["split_dir"]
        )
    )
    probes_dir = Path(args.probes_dir) if args.probes_dir else artifact_paths["probes_dir"]
    threshold_csv = Path(args.threshold_csv) if args.threshold_csv else artifact_paths["threshold_csv"]

    # Load thresholds
    print(f"\nLoading thresholds from: {threshold_csv}")
    thresholds_dict = load_thresholds_from_csv(threshold_csv)

    # Load dataset splits
    print(f"\nLoading dataset splits from: {split_dir}")
    test_task_ids = load_test_task_ids(split_dir)

    # Determine dataset config — CLI args take priority over split_summary.json
    print("\nLoading dataset configuration...")
    split_dataset_config = load_dataset_config_from_split(split_dir)
    dataset_name = (
        cfg["dataset_name"]
        if args.dataset
        else (split_dataset_config or {}).get("dataset_name", cfg["dataset_name"])
    )
    dataset_split = (
        cfg["split"]
        if args.split
        else (split_dataset_config or {}).get("split", cfg["split"])
    )
    prompt_field = (
        cfg["prompt_field"]
        if args.prompt_field
        else (split_dataset_config or {}).get("prompt_field", cfg["prompt_field"])
    )
    print(f"   Dataset:      {dataset_name}")
    print(f"   Split:        {dataset_split}")
    print(f"   Prompt field: {prompt_field}")

    print(f"\nLoading dataset: {dataset_name} (split: {dataset_split})")
    ds = load_dataset(dataset_name, split=dataset_split)

    all_tasks = [
        get_task_fields(ds[i], dataset_name, prompt_field)
        for i in range(len(ds))
    ]
    print(f"✅ Loaded {len(all_tasks)} tasks from {dataset_name}")

    # Filter to test split
    if test_task_ids is not None:
        test_tasks = filter_tasks_by_split(all_tasks, test_task_ids)
    else:
        print("⚠️  No split found — using all tasks")
        test_tasks = all_tasks

    if cfg.get("limit_tasks") is not None:
        test_tasks = test_tasks[: cfg["limit_tasks"]]
        print(f"   Limited to {len(test_tasks)} tasks")

    # Store all results
    all_results = []

    for model_id in models:
        model_family = infer_model_family(model_id)
        for feature_method in feature_methods:
            for classifier in classifiers:
                model_name_safe = model_id.replace("/", "_")
                probe_name = f"{model_name_safe}_{feature_method}_{classifier}"

                if probe_name not in thresholds_dict:
                    print(f"\n⚠️  No threshold found for {probe_name}, skipping...")
                    continue

                threshold = thresholds_dict[probe_name]

                # Exact split of strategies:
                # - Full function regeneration: SLT-only (self-correction enabled)
                # - Proactive regeneration via AdaDec: TBG-only (adaptive enabled)
                if feature_method == "TBG":
                    cfg["run_self_correction_with_uncertainty"] = False
                    cfg["run_self_correction_with_verification"] = False
                    cfg["run_adaptive_decoding"] = True
                else:
                    cfg["run_self_correction_with_uncertainty"] = True
                    cfg["run_self_correction_with_verification"] = True
                    cfg["run_adaptive_decoding"] = False

                try:
                    result = evaluate_on_humaneval_for_model(
                        model_id=model_id,
                        model_family=model_family,
                        feature_method=feature_method,
                        threshold=threshold,
                        test_tasks=test_tasks,
                        cfg=cfg,
                        hf_token=HF_TOKEN,
                        classifier=classifier,
                        probes_dir=probes_dir,
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
            if adaptive.get("error_bars"):
                ci = adaptive["error_bars"]
                print(
                    f"    Pass@1 CI ({ci.get('ci_level', 0.95):.2f}): "
                    f"[{ci['adaptive_pass_at_1_ci'][0]:.4f}, {ci['adaptive_pass_at_1_ci'][1]:.4f}]"
                )
                print(
                    f"    Improvement CI: "
                    f"[{ci['improvement_ci'][0]:+.4f}, {ci['improvement_ci'][1]:+.4f}]"
                )
            if adaptive.get("multi_trial"):
                mt = adaptive["multi_trial"]
                print(
                    f"    Multi-trial ({mt['n_trials']}): "
                    f"Pass@1={mt['adaptive_pass_at_1_mean']:.4f}±{mt['adaptive_pass_at_1_std']:.4f}"
                )

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
