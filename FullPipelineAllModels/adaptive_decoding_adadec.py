#!/usr/bin/env python3
"""
Adaptive Decoding using AdaDec's Generator (AdaFixL) + SEP Probe

This is an alternative implementation of adaptive decoding that delegates the
token-by-token generation to AdaDec's Generator class (AdaFixL mode) instead
of the custom lookahead loop in adaptive_decoding_lambda.py.

Architecture:
- SEP probe gates at the task level: should this task use adaptive decoding?
- When triggered, AdaDec's Generator handles generation with per-token entropy
  gating: at each step, if Shannon entropy of the next-token distribution
  exceeds the learned threshold, lookahead reranking is used; otherwise greedy.

AdaDec submodule: FullPipelineAllModels/AdaDec
Paper: https://arxiv.org/abs/2406.12399
"""

import os
import sys
import re
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch

from utils import (
    generate_one_sample,
    estimate_uncertainty,
    extract_code,
    get_probe_path,
)

# Re-use helpers that are identical to adaptive_decoding_lambda
from adaptive_decoding_lambda import (
    load_sep_probe,
    evaluate_completion,
    greedy_decode,
    get_config,
    get_adaptive_config,
)

# ── AdaDec submodule import ───────────────────────────────────────────────────
_ADADEC_PATH = os.path.join(os.path.dirname(__file__), "AdaDec")
if os.path.isdir(_ADADEC_PATH) and _ADADEC_PATH not in sys.path:
    sys.path.insert(0, _ADADEC_PATH)

try:
    from llm.generator import Generator as AdaDecGenerator
    _ADADEC_AVAILABLE = True
except ImportError:
    _ADADEC_AVAILABLE = False
    AdaDecGenerator = None
    print(
        "AdaDec submodule not importable. "
        "Run: git submodule update --init --recursive"
    )


# ============================================================
# Core: AdaDec-backed adaptive_decode
# ============================================================


def adaptive_decode(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    beam_size: int = 3,
    lookahead_length: int = 5,
    sep_probe: Optional[Tuple] = None,  # unused here; kept for API parity
    token_entropy_threshold: float = 3.5,
) -> Tuple[str, int, float, float]:
    """
    AdaDec-backed adaptive decoding.

    Uses AdaDec's Generator in AdaFixL mode: at each generation step, if the
    Shannon entropy of the next-token distribution exceeds token_entropy_threshold,
    lookahead beam reranking is triggered; otherwise greedy selection is used.

    The sep_probe argument is accepted for API parity with adaptive_decoding_lambda
    but is not used here — task-level gating is handled by evaluate_adaptive_decoding
    before calling this function.

    Args:
        tok: Tokenizer
        model: Language model
        prompt: Raw prompt string (chat formatting is applied internally by AdaDec)
        max_new_tokens: Maximum tokens to generate
        beam_size: Beam size for lookahead reranking
        lookahead_length: Number of lookahead steps (L from AdaDec paper)
        sep_probe: Ignored (kept for API parity)
        token_entropy_threshold: Shannon entropy threshold for triggering lookahead

    Returns:
        (generated_text, num_tokens, total_time, adaptive_ratio)
    """
    if not _ADADEC_AVAILABLE or AdaDecGenerator is None:
        raise RuntimeError(
            "AdaDec is not available. Run: git submodule update --init --recursive"
        )

    # Strip the "# Your code below:\n" suffix that the pipeline appends so that
    # AdaDec's _apply_chat_template receives a clean problem statement.
    clean_prompt = re.sub(r"\n\n#\s*Your code below[:\s]*\n?$", "", prompt).rstrip()

    generator = AdaDecGenerator(
        model=model,
        tokenizer=tok,
        model_name="unknown",   # threshold is passed explicitly below
        beam_size=beam_size,
        decoding_mode="AdaFixL",
        entropy_threshold=token_entropy_threshold,
    )

    start_time = time.time()
    generated_texts = generator.generate(
        prompt=clean_prompt,
        beam_size=beam_size,
        max_new_tokens=max_new_tokens,
        lookahead_length=lookahead_length,
        lookahead_beam_size=beam_size,
    )
    total_time = time.time() - start_time

    generated_text = generated_texts[0] if generated_texts else ""

    total_steps = generator.tradition_times + generator.lookahead_times
    adaptive_ratio = generator.lookahead_times / max(total_steps, 1)

    num_tokens = len(tok.encode(generated_text, add_special_tokens=False))

    return generated_text, num_tokens, total_time, adaptive_ratio


# ============================================================
# Evaluation (same structure as adaptive_decoding_lambda,
# but calls the AdaDec-backed adaptive_decode above)
# ============================================================

from tqdm import tqdm


def evaluate_adaptive_decoding(
    tasks: List[Dict[str, Any]],
    tok,
    model,
    cfg,
    sep_probe: Optional[Tuple] = None,
    baseline_results: Optional[Dict[str, Any]] = None,
    baseline_task_by_id: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate AdaDec-backed adaptive decoding vs greedy baseline.

    SEP probe gates at the task level (should we bother for this problem?).
    When triggered, generation is handled by AdaDec's Generator (AdaFixL mode).
    """
    n = len(tasks)
    print(f"Evaluating {n} tasks with AdaDec-backed adaptive decoding")
    if n == 0:
        return {"error": "No tasks provided"}

    use_precomputed_baseline = False
    if baseline_task_by_id is not None:
        use_precomputed_baseline = True
    elif baseline_results is not None and baseline_results.get("task_results"):
        use_precomputed_baseline = True
        baseline_task_by_id = {
            tr["task_id"]: tr
            for tr in baseline_results["task_results"]
            if "task_id" in tr
        }
    else:
        baseline_task_by_id = {}

    results = {
        "baseline_pass": 0,
        "adaptive_pass": 0,
        "baseline_latencies": [],
        "adaptive_latencies": [],
        "sequence_adaptive_latencies": [],
        "adaptive_ratio": 0,
        "task_results": [],
        "impl": "adadec",
    }

    if use_precomputed_baseline:
        results["baseline_pass"] = baseline_results.get("pass")
        if results["baseline_pass"] is None:
            results["baseline_pass"] = sum(
                1 for tr in baseline_results["task_results"] if tr.get("correct")
            )
        results["baseline_latencies"] = [
            tr.get("latency")
            for tr in baseline_results["task_results"]
            if tr.get("latency") is not None
        ]

    if cfg is None:
        cfg = get_config()

    layers = cfg.get("layers", [-3, -2, -1])

    # Resolve threshold and feature method from probe or config
    if sep_probe is not None:
        if len(sep_probe) == 4:
            _, _, threshold, feature_method = sep_probe
        else:
            _, _, threshold = sep_probe
            feature_method = cfg.get("feature_method", "SLT")
    else:
        threshold = cfg.get("uncertainty_threshold", 0.5)
        feature_method = cfg.get("feature_method", "SLT")

    token_entropy_threshold = cfg.get(
        "fallback_token_entropy_threshold", 3.5
    )

    adaptive_steps = 0

    for task in tqdm(tasks, desc="AdaDec adaptive decoding", unit="task"):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_src = task["test"]
        entry_point = task["entry_point"]
        code_prompt = task.get("code_prompt", "")

        user_prompt = prompt + "\n\n# Your code below:\n"

        # --- Baseline (greedy) ---
        if use_precomputed_baseline and task_id in baseline_task_by_id:
            btr = baseline_task_by_id[task_id]
            base_correct = btr.get("correct")
            base_latency = btr.get("latency")
        else:
            base_text, base_len, base_latency = greedy_decode(
                tok, model, user_prompt, max_new_tokens=cfg["max_new_tokens"]
            )
            base_code = extract_code(base_text)
            base_correct = evaluate_completion(
                prompt, test_src, entry_point, base_code,
                timeout_s=cfg["test_timeout_s"], code_prompt=code_prompt,
            )
            if base_correct:
                results["baseline_pass"] += 1
            results["baseline_latencies"].append(base_latency)

        start_time = time.time()

        if feature_method == "TBG":
            # TBG: check uncertainty from prompt before full generation
            _, tbg_ids, tbg_prompt_len = generate_one_sample(
                tok, model, prompt, max_new_tokens=1,
                temperature=0.0, top_p=0.95, return_ids=True,
            )
            uncertainty = estimate_uncertainty(
                tok, model, prompt, "",
                sep_probe, layers=layers,
                full_ids_cpu=tbg_ids, prompt_len=tbg_prompt_len,
            )
            print(f"TBG Uncertainty: {uncertainty:.4f}, Threshold: {threshold:.4f}")

            if uncertainty >= threshold:
                print("Triggering AdaDec adaptive decoding (TBG gate)")
                ada_text, ada_len, sequence_ada_latency, sequence_ada_ratio = (
                    adaptive_decode(
                        tok, model, user_prompt,
                        max_new_tokens=cfg["max_new_tokens"],
                        beam_size=cfg["beam_size"],
                        lookahead_length=cfg["lookahead_length"],
                        token_entropy_threshold=token_entropy_threshold,
                    )
                )
                ada_code = extract_code(ada_text)
                adaptive_steps += 1
            else:
                ada_code = generate_one_sample(
                    tok, model, prompt, max_new_tokens=cfg["max_new_tokens"],
                    temperature=0.0, top_p=0.95, return_ids=False,
                )
                ada_len = 0
                sequence_ada_latency = 0.0
                sequence_ada_ratio = 0.0
        else:
            # SLT: generate first, then check uncertainty
            ada_code, full_ids_cpu, gen_prompt_len = generate_one_sample(
                tok, model, prompt, max_new_tokens=cfg["max_new_tokens"],
                temperature=0.0, top_p=0.95, return_ids=True,
            )
            uncertainty = estimate_uncertainty(
                tok, model, prompt, ada_code, sep_probe, layers=layers,
                full_ids_cpu=full_ids_cpu, prompt_len=gen_prompt_len,
            )
            print(f"SLT Uncertainty: {uncertainty:.4f}, Threshold: {threshold:.4f}")

            if uncertainty >= threshold:
                print("Triggering AdaDec adaptive decoding (SLT gate)")
                ada_text, ada_len, sequence_ada_latency, sequence_ada_ratio = (
                    adaptive_decode(
                        tok, model, user_prompt,
                        max_new_tokens=cfg["max_new_tokens"],
                        beam_size=cfg["beam_size"],
                        lookahead_length=cfg["lookahead_length"],
                        token_entropy_threshold=token_entropy_threshold,
                    )
                )
                ada_code = extract_code(ada_text)
                adaptive_steps += 1
            else:
                ada_len = 0
                sequence_ada_latency = 0.0
                sequence_ada_ratio = 0.0

        ada_latency = time.time() - start_time

        ada_correct = evaluate_completion(
            prompt, test_src, entry_point, ada_code,
            timeout_s=cfg["test_timeout_s"], code_prompt=code_prompt,
        )
        if ada_correct:
            results["adaptive_pass"] += 1

        results["adaptive_latencies"].append(ada_latency)
        results["sequence_adaptive_latencies"].append(sequence_ada_latency)
        results["task_results"].append({
            "task_id": task_id,
            "baseline_correct": base_correct,
            "adaptive_correct": ada_correct,
            "baseline_latency": base_latency,
            "adaptive_latency": ada_latency,
            "improved": ada_correct and not base_correct,
            "degraded": base_correct and not ada_correct,
        })

    results["adaptive_ratio"] = adaptive_steps / n
    results["baseline_pass_at_1"] = results["baseline_pass"] / n
    results["adaptive_pass_at_1"] = results["adaptive_pass"] / n
    results["improvement"] = (
        results["adaptive_pass_at_1"] - results["baseline_pass_at_1"]
    )
    results["avg_baseline_latency"] = float(np.mean(results["baseline_latencies"] or [0]))
    results["avg_adaptive_latency"] = float(np.mean(results["adaptive_latencies"] or [0]))
    results["avg_sequence_adaptive_latency"] = float(
        np.mean(results["sequence_adaptive_latencies"] or [0])
    )
    results["num_improved"] = sum(1 for r in results["task_results"] if r["improved"])
    results["num_degraded"] = sum(1 for r in results["task_results"] if r["degraded"])

    adaptive_ratio = results["adaptive_ratio"]
    if adaptive_ratio < 0.01:
        print(
            f"\nWARNING: AdaDec rarely triggered ({adaptive_ratio:.2%}). "
            "Threshold may be too high."
        )
    elif adaptive_ratio > 0.9:
        print(
            f"\nWARNING: AdaDec triggered very often ({adaptive_ratio:.2%}). "
            "Threshold may be too low."
        )

    return results
