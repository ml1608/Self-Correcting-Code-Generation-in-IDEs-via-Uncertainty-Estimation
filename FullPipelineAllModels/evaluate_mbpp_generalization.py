#!/usr/bin/env python3
"""
Evaluate probe generalization to MBPP.

This runs the full pipeline on MBPP using probes/thresholds from a source dataset
(default: bigcodebench), which measures cross-dataset transfer.
"""

import argparse
import subprocess
from pathlib import Path


def get_dataset_safe_name(dataset_name: str) -> str:
    if "bigcodebench" in dataset_name.lower():
        return "bigcodebench"
    if "humaneval" in dataset_name.lower():
        return "humaneval"
    return dataset_name.replace("/", "_").replace(":", "_")


def main():
    parser = argparse.ArgumentParser(description="Run MBPP probe generalization evaluation.")
    parser.add_argument("--source_dataset", default="bigcode/bigcodebench")
    parser.add_argument("--target_dataset", default="mbpp")
    parser.add_argument("--target_split", default="test")
    parser.add_argument("--target_prompt_field", default="text")
    parser.add_argument("--models", nargs="+", default=[
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
    ])
    parser.add_argument("--feature_methods", nargs="+", default=["SLT", "TBG"])
    parser.add_argument("--classifiers", nargs="+", default=["linreg"])
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--adaptive_trials", type=int, default=1)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--output_dir", default="pipeline_results_mbpp_generalization")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    source_safe = get_dataset_safe_name(args.source_dataset)
    probes_dir = root / "Dataset+Probes" / "saved_probes" / source_safe
    threshold_csv = root / "Dataset+Probes" / "threshold_analysis_plots" / source_safe / "threshold_recommendations.csv"
    split_dir = root / "Dataset+Probes" / "DatasetSplit" / get_dataset_safe_name(args.target_dataset)

    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "run_pipeline_all_models.py"),
        "--dataset", args.target_dataset,
        "--split", args.target_split,
        "--prompt_field", args.target_prompt_field,
        "--probes_dir", str(probes_dir),
        "--threshold_csv", str(threshold_csv),
        "--split_dir", str(split_dir),
        "--output_dir", args.output_dir,
        "--adaptive_trials", str(args.adaptive_trials),
        "--bootstrap_samples", str(args.bootstrap_samples),
        "--ci_level", "0.95",
    ]
    if args.limit_tasks is not None:
        cmd.extend(["--limit_tasks", str(args.limit_tasks)])
    cmd.extend(["--models", *args.models])
    cmd.extend(["--feature_methods", *args.feature_methods])
    cmd.extend(["--classifiers", *args.classifiers])

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
