#!/usr/bin/env python3
"""
Adaptive Decoding using SEP Probe
Uses trained Semantic Entropy Probe to predict high semantic entropy
and triggers adaptive decoding (beam search with lookahead) when needed.
"""

import os
import re
import io
import json
import time
import math
import signal
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass
from tqdm import tqdm

from utils import (
    generate_one_sample,
    build_chat_text,
    estimate_uncertainty,
    extract_features_multi_method,
    extract_code,
    get_probe_path,
)

# ============================================================
# Configuration
# ============================================================
 

def get_config():
    """
    Get configuration for ADADEC-style adaptive decoding.
    Uses TBG probe to detect uncertainty early, then uses lookahead scoring.
    """
    return {
        # To use DeepSeek: "deepseek-ai/deepseek-coder-1.3b-instruct"
        # To use Qwen: "Qwen/Qwen2.5-Coder-3B-Instruct"
        # To use Llama: "meta-llama/Llama-3.2-3B-Instruct"
        # Set the model_id as needed below:
        # "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        "dataset_name": "openai_humaneval",
        "split": "test",
        "limit_tasks": 10,  # Set to a number for testing, or None for full 164
        # Adaptive decoding parameters (ADADEC-style)
        "max_new_tokens": 256,
        "beam_size": 3,  # Top-K candidates for lookahead (B from ADADEC paper)
        "lookahead_length": 5,  # Fixed lookahead length (L from ADADEC paper)
        # SEP probe settings
        "probe_path": "sep_slt_runs/meta-llama_Llama-3.2-3B-Instruct",  # Path to trained probe
        "use_sep_probe": True,  # Use SEP probe to trigger adaptive decoding
        "uncertainty_threshold": None,  # Auto-loaded from probe (None = use probe's recommended threshold)
        "fallback_token_entropy_threshold": 0.3,  # Fallback if probe not available 
        # Feature extraction (must match training config)
        "layers": [-3, -2, -1],
        # Evaluation
        "test_timeout_s": 10,
        "seed": 42,
        "feature_method": "TBG",
    }


# Alias for compatibility
get_adaptive_config = get_config


# ============================================================
# Load SEP Probe
# ============================================================


def load_sep_probe(
    probe_dir: str,
    threshold_override: Optional[float] = None,
    feature_method: str = "SLT",
):
    """
    Load trained SEP probe and scaler.

    Args:
        probe_dir: Directory containing probe.pkl and probe_metadata.json
        threshold_override: Optional threshold to override the one in probe metadata
        feature_method: Feature method used (SLT or TBG) - returned as part of tuple

    Returns:
        (scaler, clf, threshold, feature_method) tuple
    """
    probe_pkl_path = os.path.join(probe_dir, "probe.pkl")
    probe_json_path = os.path.join(probe_dir, "probe_metadata.json")

    # Try loading from pickle first (preferred)
    if os.path.exists(probe_pkl_path):
        with open(probe_pkl_path, "rb") as f:
            probe_data = pickle.load(f)
        scaler = probe_data["scaler"]
        clf = probe_data["classifier"]

        # Get feature method from metadata if available
        if os.path.exists(probe_json_path):
            with open(probe_json_path, "r") as f:
                probe_info = json.load(f)
            feature_method = probe_info.get("feature_method", feature_method)
            classifier_type = probe_info.get("classifier", "unknown")
        else:
            classifier_type = "unknown"

        # Use threshold override if provided, otherwise try to get from probe data
        if threshold_override is not None:
            threshold = threshold_override
            print(f"✅ Loaded SEP probe: using override threshold={threshold:.4f}")
        elif "recommended_threshold" in probe_data:
            threshold = probe_data["recommended_threshold"]
            print(
                f"✅ Loaded SEP probe: using recommended threshold={threshold:.4f} (from probe training)"
            )
        elif "threshold" in probe_data:
            threshold = probe_data["threshold"]
            print(
                f"✅ Loaded SEP probe: using semantic entropy threshold={threshold:.4f} (fallback)"
            )
        else:
            threshold = 0.5  # Default fallback
            print(f"⚠️  No threshold found in probe, using default={threshold:.4f}")

        print(
            f"✅ Loaded SEP probe from pickle: {classifier_type}, feature_method={feature_method}, threshold={threshold:.4f}"
        )
        return scaler, clf, threshold, feature_method

    # Fallback: try to reconstruct from JSON (limited support)
    if os.path.exists(probe_json_path):
        print(
            f"⚠️  Pickle file not found, attempting to load from JSON (limited support)"
        )
        with open(probe_json_path, "r") as f:
            probe_info = json.load(f)

        # Load scaler parameters
        scaler = StandardScaler()
        scaler.mean_ = np.array(probe_info["scaler_mean"])
        scaler.scale_ = np.array(probe_info["scaler_scale"])

        # For classifier, we can't fully reconstruct from JSON
        # This is a fallback that won't work for RandomForest
        classifier_type = probe_info.get("classifier", "unknown")
        feature_method = probe_info.get("feature_method", feature_method)
        print(
            f"⚠️  Cannot fully reconstruct {classifier_type} from JSON. Please use probe.pkl"
        )
        return None, None, None, feature_method

    print(f"⚠️  Probe not found at {probe_dir}")
    return None, None, None, feature_method


@torch.inference_mode()
def get_next_token_with_features(tok, model, input_ids, layers, prompt_len=None):
    """
    Get next token distribution and extract TBG features (ADADEC-style).

    TBG = Token Before Generation = last token in current sequence
    This is extracted at each step during generation.

    Returns:
        - logits: probability distribution over vocabulary
        - tbg_features: features from last token in current sequence
    """
    # Forward pass
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    logits = outputs.logits[0, -1, :]  # Last token logits

    # Extract TBG features (Token Before Generation = last token in current sequence)
    # At each generation step, this is the last token we've generated so far
    token_idx = input_ids.shape[1] - 1  # Last token in current sequence

    features = []
    for layer in layers:
        hs = outputs.hidden_states[layer]
        # Ensure token_idx is valid
        if token_idx >= hs.shape[1]:
            token_idx = hs.shape[1] - 1
        if token_idx < 0:
            token_idx = 0
        features.append(hs[0, token_idx, :].float().detach().cpu().numpy())

    tbg_features = np.concatenate(features)

    return logits, tbg_features


@torch.inference_mode()
def lookahead_score_token(tok, model, input_ids, candidate_token_id, lookahead_length):
    """
    ADADEC Lookahead Scoring (Section III-C of paper)

    Score a candidate token by:
    1. Append candidate token to sequence
    2. Generate next L tokens greedily (lookahead trajectory)
    3. Compute average log-probability over the trajectory

    Args:
        tok: tokenizer
        model: language model
        input_ids: current input sequence [1, seq_len]
        candidate_token_id: token to score
        lookahead_length: how many steps to lookahead (L from paper)

    Returns:
        score: average log-probability of trajectory
    """
    # Start trajectory with candidate token
    trajectory_ids = torch.cat(
        [input_ids, torch.tensor([[candidate_token_id]]).to(input_ids.device)], dim=1
    )

    # Track log probabilities
    log_probs = []

    # Get log-prob of candidate token itself
    outputs = model(input_ids, use_cache=True)
    logits = outputs.logits[0, -1, :]
    log_prob_dist = torch.log_softmax(logits, dim=0)
    log_probs.append(log_prob_dist[candidate_token_id].item())

    # Lookahead: generate next L tokens greedily
    for step in range(lookahead_length):
        outputs = model(trajectory_ids, use_cache=False)
        logits = outputs.logits[0, -1, :]

        # Greedy selection for lookahead
        next_token_id = logits.argmax().item()

        # Get log-probability
        log_prob_dist = torch.log_softmax(logits, dim=0)
        log_probs.append(log_prob_dist[next_token_id].item())

        # Append to trajectory
        trajectory_ids = torch.cat(
            [trajectory_ids, torch.tensor([[next_token_id]]).to(trajectory_ids.device)],
            dim=1,
        )

        # Stop if EOS
        if next_token_id == tok.eos_token_id:
            break

    # Average log-probability (geometric mean, as in paper Section III-C)
    avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else -float("inf")

    return avg_log_prob


@torch.inference_mode()
def extract_slt_vec_multi_layer(
    tok, model, full_ids_cpu: torch.Tensor, layers: list = [-3, -2, -1]
):
    """Extract SLT features from multiple layers and concatenate them (backward compatibility)."""
    # For backward compatibility, we need prompt_len, but we'll use -2 for SLT
    # This is a simplified version that assumes we're extracting from the current state
    if full_ids_cpu.dim() == 1:
        full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
    else:
        full_ids = full_ids_cpu.to(model.device)
    out = model(full_ids, output_hidden_states=True, use_cache=False)

    features = []
    for layer in layers:
        hs = out.hidden_states[layer]
        # SLT uses second-to-last token
        token_idx = -2
        if token_idx < 0:
            token_idx = hs.shape[1] + token_idx
        if token_idx >= hs.shape[1]:
            token_idx = hs.shape[1] - 1
        if token_idx < 0:
            token_idx = 0
        features.append(hs[0, token_idx, :].float().detach().cpu().numpy())

    return np.concatenate(features)


# ============================================================
# Decoding Functions
# ============================================================


@torch.inference_mode()
def greedy_decode(
    tok, model, prompt: str, max_new_tokens: int = 256
) -> Tuple[str, int, float]:
    """Standard greedy decoding."""
    chat_text = build_chat_text(tok, prompt)
    input_ids = tok(chat_text, return_tensors="pt").input_ids.to(model.device)
    output_ids = input_ids.clone()
    start = time.time()

    for _ in range(max_new_tokens):
        logits = model(output_ids).logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1))
        output_ids = torch.cat(
            [output_ids, torch.tensor([[next_id]], device=model.device)], dim=1
        )
        if next_id == tok.eos_token_id:
            break

    latency = time.time() - start
    gen_text = tok.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return gen_text, (output_ids.shape[1] - input_ids.shape[1]), latency


# generate entire solution using adaptive decoding
@torch.inference_mode()
def adaptive_generate_one_token(
    tok, model, input_ids, probe_data, threshold, cfg, layers, verbose=False
):
    """
    ADADEC: Generate next token with adaptive decoding

    This implements the full pause-then-rerank mechanism from the paper:
    1. Get probability distribution and TBG features
    2. Calculate uncertainty from probe
    3. If uncertainty > threshold: PAUSE and RERANK using lookahead
    4. If uncertainty <= threshold: use greedy (top-1 token)

    Returns:
        - next_token_id: selected token
        - used_adaptive: whether adaptive decoding was triggered
        - uncertainty_score: uncertainty score from probe
    """
    # Step 1: Get next token distribution and TBG features
    logits, tbg_features = get_next_token_with_features(tok, model, input_ids, layers)

    # Step 2: Get uncertainty score from probe
    scaler = probe_data["scaler"]
    classifier = probe_data["classifier"]
    features_scaled = scaler.transform(tbg_features.reshape(1, -1))
    uncertainty_score = classifier.predict_proba(features_scaled)[0, 1]

    if verbose:
        print(
            f"    Probe uncertainty: {uncertainty_score:.4f}, Threshold: {threshold:.4f}"
        )

    # Step 3: Decide whether to use adaptive decoding
    if uncertainty_score > threshold:
        # UNCERTAIN: Pause and rerank using lookahead
        if verbose:
            print(f"    → ADAPTIVE DECODING triggered!")

        # Get top-B candidates
        top_k_values, top_k_indices = torch.topk(logits, k=cfg["lookahead_beam_size"])

        # Score each candidate using lookahead
        candidate_scores = []
        for i in range(cfg["lookahead_beam_size"]):
            candidate_id = top_k_indices[i].item()
            score = lookahead_score_token(
                tok, model, input_ids, candidate_id, cfg["lookahead_length"]
            )
            candidate_scores.append((candidate_id, score))
            if verbose:
                token_text = tok.decode([candidate_id])
                print(f"      Candidate {i+1}: '{token_text}' → score={score:.4f}")

        # Select best candidate
        best_token_id = max(candidate_scores, key=lambda x: x[1])[0]
        if verbose:
            print(f"    → Selected: '{tok.decode([best_token_id])}'")

        return best_token_id, True, uncertainty_score

    else:
        # CONFIDENT: Use greedy (top-1 token)
        if verbose:
            print(f"    → GREEDY (confident)")
        best_token_id = logits.argmax().item()
        return best_token_id, False, uncertainty_score


@torch.inference_mode()
def adaptive_decode(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    beam_size: int = 3,
    lookahead_length: int = 5,
    sep_probe: Optional[Tuple] = None,  # (scaler, clf, threshold, feature_method)
    token_entropy_threshold: float = 3.5,
) -> Tuple[str, int, float, float]:
    """
    ADADEC-style adaptive decoding: uses TBG/SLT probe to detect uncertainty early.
    When uncertain: use lookahead scoring (top-K candidates, lookahead L steps, pick best).
    When confident: use greedy.

    Returns:
        - generated_text: generated code
        - num_tokens: number of tokens generated
        - total_time: total generation time
        - adaptive_ratio: fraction of steps that used adaptive decoding
    """
    chat_text = build_chat_text(tok, prompt)
    input_ids = tok(chat_text, return_tensors="pt").input_ids.to(model.device)
    output_ids = input_ids.clone()
    start_time = time.time()

    # Track adaptive decisions
    adaptive_steps = 0
    total_steps = 0
    all_uncertainties = []

    # Prepare probe data and config
    if sep_probe is not None:
        if len(sep_probe) == 4:
            scaler, clf, threshold, feature_method = sep_probe
        else:
            scaler, clf, threshold = sep_probe
            feature_method = "TBG"  # Default to TBG for ADADEC

        probe_data = {"scaler": scaler, "classifier": clf}
        cfg = {
            "lookahead_beam_size": beam_size,
            "lookahead_length": lookahead_length,
        }
        layers = [-3, -2, -1]
    else:
        probe_data = None
        threshold = None
        cfg = None
        layers = None

    # Generate tokens one by one
    for step in range(max_new_tokens):
        if probe_data is not None:
            # ADADEC: Use probe to detect uncertainty and decide
            next_token_id, used_adaptive, uncertainty = adaptive_generate_one_token(
                tok,
                model,
                output_ids,
                probe_data,
                threshold,
                cfg,
                layers,
                verbose=(step < 3),  # Verbose for first 3 steps
            )

            if used_adaptive:
                adaptive_steps += 1
            total_steps += 1
            all_uncertainties.append(uncertainty)
        else:
            # Fallback: use token entropy
            outputs = model(output_ids, output_hidden_states=True, use_cache=True)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            p = probs[0].detach().float().cpu().numpy()
            entropy = -float(np.sum(p * np.log(p + 1e-10)))
            use_adaptive = entropy > token_entropy_threshold

            if use_adaptive:
                # Use lookahead scoring
                topk = torch.topk(probs, beam_size, dim=-1)
                candidate_tokens = topk.indices[0]

                best_score = -float("inf")
                best_token = None

                for token in candidate_tokens:
                    score = lookahead_score_token(
                        tok, model, output_ids, token.item(), lookahead_length
                    )
                    if score > best_score:
                        best_score = score
                        best_token = token.item()

                next_token_id = (
                    best_token if best_token is not None else int(torch.argmax(probs))
                )
                adaptive_steps += 1
            else:
                next_token_id = int(torch.argmax(probs))

            total_steps += 1

        # Append token
        output_ids = torch.cat(
            [output_ids, torch.tensor([[next_token_id]]).to(output_ids.device)], dim=1
        )

        # Stop if EOS
        if next_token_id == tok.eos_token_id:
            break

    total_time = time.time() - start_time

    # Decode generated text
    prompt_len = tok(chat_text, return_tensors="pt").input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:]
    generated_text = tok.decode(generated_ids, skip_special_tokens=False)

    sequence_adaptive_ratio = adaptive_steps / max(total_steps, 1)

    return (
        generated_text,
        (output_ids.shape[1] - input_ids.shape[1]),
        total_time,
        sequence_adaptive_ratio,
    )


# ============================================================
# Code Evaluation
# ============================================================

def _run_test_with_timeout(
    module_src: str, entry_point: str, timeout_seconds: int = 10
):
    """Run HumanEval test with timeout."""
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


def evaluate_completion(
    prompt_src: str, test_src: str, entry_point: str, code: str, timeout_s: int = 10
) -> bool:
    """Evaluate if generated code passes HumanEval tests."""
    if not code:
        return False
    module_src = prompt_src + "\n" + code + "\n\n" + test_src
    ok, _ = _run_test_with_timeout(module_src, entry_point, timeout_seconds=timeout_s)
    return ok


# ============================================================
# Main Evaluation
# ============================================================


def evaluate_adaptive_decoding(
    tasks: List[Dict[str, Any]],
    tok,
    model,
    cfg,
    sep_probe: Optional[Tuple] = None,
    baseline_results: Optional[Dict[str, Any]] = None,
    baseline_task_by_id: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate adaptive decoding vs greedy baseline on HumanEval."""

    n = len(tasks)
    print(f"Evaluating {n} tasks")
    if n == 0:
        return {"error": "No tasks provided"}

    use_precomputed_baseline = False
    if baseline_task_by_id is not None:
        use_precomputed_baseline = True
    elif baseline_results is not None and baseline_results.get("task_results"):
        use_precomputed_baseline = True
        baseline_task_by_id = {
            task_result["task_id"]: task_result
            for task_result in baseline_results["task_results"]
            if "task_id" in task_result
        }
    else:
        baseline_task_by_id = {}

    results = {
        "baseline_pass": 0,
        "adaptive_pass": 0,
        "baseline_latencies": [],
        "adaptive_latencies": [],
        "sequence_adaptive_latencies": [],
        "adaptive_ratio": 0,  # Fraction of steps that used adaptive decoding
        "task_results": [],
    }

    if use_precomputed_baseline:
        results["baseline_pass"] = baseline_results.get("pass")
        if results["baseline_pass"] is None:
            results["baseline_pass"] = sum(
                1
                for task_result in baseline_results["task_results"]
                if task_result.get("correct")
            )
        results["baseline_latencies"] = [
            task_result.get("latency")
            for task_result in baseline_results["task_results"]
            if task_result.get("latency") is not None
        ]

    adaptive_steps = 0

    for task in tqdm(tasks, desc="Evaluating adaptive decoding", unit="task"):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_src = task["test"]
        entry_point = task["entry_point"]

        user_prompt = prompt + "\n\n# Your code below:\n"

        if use_precomputed_baseline and task_id in baseline_task_by_id:
            baseline_task = baseline_task_by_id[task_id]
            base_correct = baseline_task.get("correct")
            base_latency = baseline_task.get("latency")
        else:
            # Baseline: greedy decode
            base_text, base_len, base_latency = greedy_decode(
                tok, model, user_prompt, max_new_tokens=cfg["max_new_tokens"]
            )
            base_code = extract_code(base_text)
            base_correct = evaluate_completion(
                prompt,
                test_src,
                entry_point,
                base_code,
                timeout_s=cfg["test_timeout_s"],
            )

            if base_correct:
                results["baseline_pass"] += 1
            results["baseline_latencies"].append(base_latency)

        # generate initial solution
        # use SEP probe on the initial solution to estimate uncertainty
        # trigger adaptive decoding only when probe indicates high uncertainty
        # adaptive ratio is the fraction of tasks that used adaptive decoding

        if cfg is None:
            cfg = get_config()

        layers = cfg.get("layers", [-3, -2, -1])
        

        # Auto-load threshold from probe if not set in config
        if cfg["uncertainty_threshold"] is None and sep_probe is not None:
            # sep_probe can be (scaler, clf, threshold) or (scaler, clf, threshold, feature_method)
            if len(sep_probe) == 4:
                _, _, threshold, feature_method = sep_probe
            else:
                _, _, threshold = sep_probe
                feature_method = cfg.get("feature_method", "SLT")
            print(f"  [INFO] Using probe's recommended threshold: {threshold:.4f}")
        else:
            threshold = cfg.get("uncertainty_threshold", 0.5)
            feature_method = cfg.get("feature_method", "SLT")
        
        start_time = time.time()

        # TBG OPTIMIZATION: For TBG features, we can estimate uncertainty from the prompt BEFORE
        # generating any code. This saves latency when adaptive decoding is triggered.
        # For SLT features, we still need to generate code first (features depend on generation).
        
        if feature_method == "TBG":
            # TBG OPTIMIZATION: Generate just 1 token, then extract TBG features
            # This gives the model a chance to "commit" to a direction while still
            # being fast enough to save latency when adaptive decoding is triggered
            _, tbg_ids, tbg_prompt_len = generate_one_sample(
                tok,
                model,
                prompt,
                max_new_tokens=1,  # Just one token for early uncertainty check
                temperature=0.0,
                top_p=0.95,
                return_ids=True,
            )
            
            # Extract TBG features (from last prompt token, before the generated token)
            uncertainty = estimate_uncertainty(
                tok, model, prompt, "",
                sep_probe, layers=layers,
                full_ids_cpu=tbg_ids,
                prompt_len=tbg_prompt_len,
            )
            
            print(f"TBG Uncertainty (1-token): {uncertainty:.4f}, Threshold: {threshold:.4f}")
            
            if uncertainty >= threshold:
                # High uncertainty - trigger adaptive decoding (skip rest of greedy)
                print("Triggering adaptive decoding (after 1-token check)")
                ada_text, ada_len, sequence_ada_latency, sequence_ada_ratio = (
                    adaptive_decode(
                        tok,
                        model,
                        user_prompt,
                        max_new_tokens=cfg["max_new_tokens"],
                        beam_size=cfg["beam_size"],
                        lookahead_length=cfg["lookahead_length"],
                        token_entropy_threshold=cfg["fallback_token_entropy_threshold"],
                    )
                )
                ada_code = extract_code(ada_text)
                adaptive_steps += 1
            else:
                # Low uncertainty - confident, continue with greedy decoding
                ada_code = generate_one_sample(
                    tok,
                    model,
                    prompt,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=0.0,
                    top_p=0.95,
                    return_ids=False,
                )
                ada_len = 0
                sequence_ada_latency = 0
                sequence_ada_ratio = 0
        else:
            # SLT: Need to generate code first, then check uncertainty
            # (SLT features depend on the generated sequence)
            ada_code, full_ids_cpu, gen_prompt_len = generate_one_sample(
                tok,
                model,
                prompt,
                max_new_tokens=cfg["max_new_tokens"],
                temperature=0.0,
                top_p=0.95,
                return_ids=True,
            )
            
            # Estimate uncertainty using SLT features (second-to-last token of full sequence)
            uncertainty = estimate_uncertainty(
                tok, model, prompt, ada_code, sep_probe, layers=layers,
                full_ids_cpu=full_ids_cpu,
                prompt_len=gen_prompt_len,
            )
            
            print(f"SLT Uncertainty: {uncertainty:.4f}, Threshold: {threshold:.4f}")

            if uncertainty >= threshold:
                # High uncertainty - regenerate with adaptive decoding
                print("Triggering adaptive decoding")
                ada_text, ada_len, sequence_ada_latency, sequence_ada_ratio = (
                    adaptive_decode(
                        tok,
                        model,
                        user_prompt,
                        max_new_tokens=cfg["max_new_tokens"],
                        beam_size=cfg["beam_size"],
                        lookahead_length=cfg["lookahead_length"],
                        token_entropy_threshold=cfg["fallback_token_entropy_threshold"],
                    )
                )
                ada_code = extract_code(ada_text)
                adaptive_steps += 1
            else:
                # Low uncertainty - keep the greedy result
                ada_len = 0
                sequence_ada_latency = 0
                sequence_ada_ratio = 0

        ada_latency = time.time() - start_time

        ada_correct = evaluate_completion(
            prompt, test_src, entry_point, ada_code, timeout_s=cfg["test_timeout_s"]
        )

        if ada_correct:
            results["adaptive_pass"] += 1
        results["adaptive_latencies"].append(ada_latency)
        results["sequence_adaptive_latencies"].append(sequence_ada_latency)
        results["task_results"].append(
            {
                "task_id": task_id,
                "baseline_correct": base_correct,
                "adaptive_correct": ada_correct,
                "baseline_latency": base_latency,
                "adaptive_latency": ada_latency,
                "improved": ada_correct and not base_correct,
                "degraded": base_correct and not ada_correct,
            }
        )
    results["adaptive_ratio"] = adaptive_steps / n

    # Compute summary statistics
    results["baseline_pass_at_1"] = results["baseline_pass"] / n
    results["adaptive_pass_at_1"] = results["adaptive_pass"] / n
    results["improvement"] = (
        results["adaptive_pass_at_1"] - results["baseline_pass_at_1"]
    )
    results["avg_baseline_latency"] = float(np.mean(results["baseline_latencies"]))
    results["avg_adaptive_latency"] = float(np.mean(results["adaptive_latencies"]))
    results["avg_sequence_adaptive_latency"] = float(
        np.mean(results["sequence_adaptive_latencies"])
    )

    # Count improvements/degradations
    results["num_improved"] = sum(1 for r in results["task_results"] if r["improved"])
    results["num_degraded"] = sum(1 for r in results["task_results"] if r["degraded"])

    # DEBUG: Analyze adaptive ratios
    adaptive_ratio = results["adaptive_ratio"]
    if adaptive_ratio < 0.01:
        print(
            f"\n⚠️  WARNING: Adaptive decoding rarely triggered (avg ratio: {adaptive_ratio:.2%})"
        )
        print(f"   This suggests the threshold may be too high")
        print(f"   Check probe training output for recommended threshold")
    elif adaptive_ratio > 0.9:
        print(
            f"\n⚠️  WARNING: Adaptive decoding triggered too often (avg ratio: {adaptive_ratio:.2%})"
        )
        print(f"   This suggests the threshold (0.3) may be too low")

    return results


# ============================================================
# Model Loading
# ============================================================


def load_model(model_id: str, hf_token: str | None):
    """Load model and tokenizer."""
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
        if (
            "gated repo" in msg.lower()
            or "401" in msg
            or "403" in msg
            or "unauthorized" in msg.lower()
        ):
            print(f"[SKIP] Unauthorized / gated model: {model_id}\n  {e}")
            return None, None
        raise


# ============================================================
# Main
# ============================================================


def main():
    print("=== Adaptive Decoding with SEP Probe ===")
    cfg = get_config()

    # HF token
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

    # Load dataset
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    tasks = [ds[i] for i in range(len(ds))]

    if cfg["limit_tasks"] is not None:
        tasks = tasks[: cfg["limit_tasks"]]
        print(f"\nUsing limited HumanEval tasks: {len(tasks)}")
    else:
        print(f"\nUsing full HumanEval tasks: {len(tasks)}")

    # Load model
    print(f"\nLoading model: {cfg['model_id']}")
    tok, model = load_model(cfg["model_id"], hf_token=HF_TOKEN)
    if tok is None:
        print("[ERROR] Could not load model")
        return

    # Load SEP probe
    sep_probe = None
    if cfg["use_sep_probe"]:
        probe_dir = get_probe_path(cfg["model_id"], cfg["feature_method"])
        scaler, clf, threshold, feature_method = load_sep_probe(probe_dir, feature_method=cfg["feature_method"])
        if scaler is not None:
            sep_probe = (scaler, clf, threshold, feature_method)
            print(f"✅ Using SEP probe from {probe_dir}")
        else:
            print(f"⚠️  SEP probe not found, using token entropy fallback")
    else:
        print("Using token entropy threshold (SEP probe disabled)")

    # Evaluate
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print(f"Beam size: {cfg['beam_size']}")
    print(f"Lookahead length: {cfg['lookahead_length']}")
    print("=" * 80)

    results = evaluate_adaptive_decoding(tasks, tok, model, cfg, sep_probe=sep_probe)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(
        f"Baseline (Greedy) Pass@1: {results['baseline_pass_at_1']:.4f} ({results['baseline_pass']}/{len(tasks)})"
    )
    print(
        f"Adaptive Pass@1:         {results['adaptive_pass_at_1']:.4f} ({results['adaptive_pass']}/{len(tasks)})"
    )
    print(f"Improvement:             {results['improvement']:+.4f}")
    print(f"\nTasks improved:  {results['num_improved']}")
    print(f"Tasks degraded:  {results['num_degraded']}")
    print(f"\nAvg baseline latency: {results['avg_baseline_latency']:.3f}s")
    print(f"Avg adaptive latency:  {results['avg_adaptive_latency']:.3f}s")
    print(
        f"Avg sequence adaptive latency:  {results['avg_sequence_adaptive_latency']:.3f}s"
    )
    print(
        f"Avg adaptive ratio:    {results['adaptive_ratio']:.2%} (fraction of steps using adaptive)"
    )

    # Save results
    output_file = "adaptive_decoding_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
