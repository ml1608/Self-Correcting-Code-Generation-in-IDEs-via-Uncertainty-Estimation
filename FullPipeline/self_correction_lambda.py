#!/usr/bin/env python3
"""
Self-Correction Module

This file implements uncertainty-guided self-correction:

1. Load a trained probe to estimate uncertainty
2. Generate code and compute uncertainty score
3. If uncertainty exceeds threshold, trigger correction
4. Apply correction strategy (resampling, adaptive decoding)
5. Repeat until confident or max attempts reached
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
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass
from tqdm import tqdm

# Import functions from adaptive_decoding_lambda
# If import fails, functions are defined below as fallback
try:
    from adaptive_decoding_lambda import (
        load_sep_probe,
        extract_slt_vec_multi_layer,
        build_chat_text,
        greedy_decode,
        adaptive_decode,
        extract_code,
        evaluate_completion,
        load_model,
    )
    _IMPORTED_FUNCTIONS = True
except ImportError:
    _IMPORTED_FUNCTIONS = False
    # Functions will be defined below

# ============================================================
# Configuration
# ============================================================

def get_config():
    return {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        
        "dataset_name": "openai_humaneval",
        "split": "test",
        "limit_tasks": None,  # Set to a number for testing, or None for full 164
        
        # Self-correction parameters
        "uncertainty_threshold": None,  # Auto-loaded from probe (None = use probe's recommended threshold)
        "max_attempts": 2,  # Maximum correction attempts (reduced from 3 for speed)
        "correction_strategy": "resample",  # Options: "adaptive", "resample", "both" (resample is faster)
        
        # Generation parameters
        "max_new_tokens": 256,
        "initial_temperature": 0.0,  # Greedy for initial generation
        "correction_temperature": 0.3,  # Slightly stochastic for corrections
        
        # Adaptive decoding parameters (if using adaptive strategy)
        "beam_size": 3,
        "lookahead_length": 5,
        
        # Resampling parameters (if using resample strategy)
        "resample_temperature": 0.7,
        "resample_top_p": 0.95,
        "num_resamples": 2,  # Number of resamples to try (reduced from 3 for speed)
        
        # SEP probe settings
        "probe_path": "sep_slt_runs/meta-llama_Llama-3.2-3B-Instruct",
        "use_sep_probe": True,
        
        # Feature extraction (must match training config)
        "layers": [-3, -2, -1],
        
        # Evaluation
        "test_timeout_s": 10,
        "seed": 42,
    }

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function so that it passes the tests. "
    "Return only Python code, no explanation."
)

# ============================================================
# Helper Functions (if not imported)
# ============================================================

if not _IMPORTED_FUNCTIONS:
    # Define functions here if import failed
    def load_sep_probe(probe_dir: str):
        """Load trained SEP probe and scaler."""
        probe_pkl_path = os.path.join(probe_dir, "probe.pkl")
        if os.path.exists(probe_pkl_path):
            with open(probe_pkl_path, "rb") as f:
                probe_data = pickle.load(f)
            scaler = probe_data["scaler"]
            clf = probe_data["classifier"]
            
            # Use recommended_threshold if available (for inference), otherwise fall back to threshold
            if "recommended_threshold" in probe_data:
                threshold = probe_data["recommended_threshold"]
                print(f"✅ Loaded SEP probe: using recommended threshold={threshold:.4f} (auto-loaded from probe training)")
            elif "threshold" in probe_data:
                threshold = probe_data["threshold"]
                print(f"✅ Loaded SEP probe: using semantic entropy threshold={threshold:.4f} (fallback)")
            else:
                threshold = 0.5  # Default fallback
                print(f"⚠️  No threshold found in probe, using default={threshold:.4f}")
            
            return scaler, clf, threshold
        print(f"⚠️  Probe not found at {probe_dir}")
        return None, None, None
    
    @torch.inference_mode()
    def extract_slt_vec_multi_layer(tok, model, full_ids_cpu: torch.Tensor, layers: list = [-3, -2, -1]):
        """Extract features from multiple layers and concatenate them."""
        full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
        out = model(full_ids, output_hidden_states=True, use_cache=False)
        features = []
        for layer in layers:
            hs = out.hidden_states[layer]
            features.append(hs[0, -1, :].float().detach().cpu().numpy())
        return np.concatenate(features)
    
    def build_chat_text(tok, user_prompt: str):
        """Build chat-formatted text for Llama."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(tok, "chat_template", None) not in (None, ""):
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return f"[SYSTEM] {SYSTEM_PROMPT}\n[USER] {user_prompt}\n[ASSISTANT]\n"
    
    @torch.inference_mode()
    def greedy_decode(tok, model, prompt: str, max_new_tokens: int = 256) -> Tuple[str, int, float]:
        """Standard greedy decoding."""
        chat_text = build_chat_text(tok, prompt)
        input_ids = tok(chat_text, return_tensors="pt").input_ids.to(model.device)
        output_ids = input_ids.clone()
        start = time.time()
        for _ in range(max_new_tokens):
            logits = model(output_ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1))
            output_ids = torch.cat([output_ids, torch.tensor([[next_id]], device=model.device)], dim=1)
            if next_id == tok.eos_token_id:
                break
        latency = time.time() - start
        gen_text = tok.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        return gen_text, (output_ids.shape[1] - input_ids.shape[1]), latency
    
    @torch.inference_mode()
    def adaptive_decode(tok, model, prompt: str, max_new_tokens: int = 256,
                       beam_size: int = 3, lookahead_length: int = 5,
                       sep_probe: Optional[Tuple] = None,
                       token_entropy_threshold: float = 3.5) -> Tuple[str, int, float, float]:
        """Adaptive decoding with beam search and lookahead."""
        chat_text = build_chat_text(tok, prompt)
        input_ids = tok(chat_text, return_tensors="pt").input_ids.to(model.device)
        output_ids = input_ids.clone()
        start_time = time.time()
        adaptive_decisions = 0
        total_decisions = 0
        for step in range(max_new_tokens):
            outputs = model(output_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            use_adaptive = False
            if sep_probe is not None:
                scaler, clf, threshold = sep_probe
                full_ids_cpu = output_ids.detach().cpu()
                feat = extract_slt_vec_multi_layer(tok, model, full_ids_cpu, layers=[-3, -2, -1])
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                prob_high_entropy = clf.predict_proba(feat_scaled)[0, 1]
                use_adaptive = prob_high_entropy > 0.5
            else:
                p = probs[0].detach().float().cpu().numpy()
                entropy = -float(np.sum(p * np.log(p + 1e-10)))
                use_adaptive = entropy > token_entropy_threshold
            total_decisions += 1
            if use_adaptive:
                adaptive_decisions += 1
                topk = torch.topk(probs, beam_size, dim=-1)
                candidate_tokens = topk.indices[0]
                candidate_probs = topk.values[0]
                best_score = -float("inf")
                best_token = None
                for token, token_prob in zip(candidate_tokens, candidate_probs):
                    sim_ids = torch.cat([output_ids, token.view(1, 1)], dim=1)
                    score = math.log(float(token_prob) + 1e-10)
                    for _ in range(lookahead_length):
                        sim_out = model(sim_ids)
                        sim_logits = sim_out.logits[:, -1, :]
                        sim_probs = torch.nn.functional.softmax(sim_logits, dim=-1)
                        nxt = torch.argmax(sim_probs, dim=-1)
                        nxt_prob = sim_probs[0, nxt]
                        score += math.log(float(nxt_prob) + 1e-10)
                        sim_ids = torch.cat([sim_ids, nxt.view(1, 1)], dim=1)
                        if tok.decode(int(nxt)) == "\n":
                            break
                    traj_len = sim_ids.shape[1] - output_ids.shape[1]
                    avg_log_prob = score / max(traj_len, 1)
                    if avg_log_prob > best_score:
                        best_score = avg_log_prob
                        best_token = int(token)
                next_token_id = best_token if best_token is not None else int(torch.argmax(probs))
            else:
                next_token_id = int(torch.argmax(probs))
            output_ids = torch.cat([output_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)
            if next_token_id == tok.eos_token_id:
                break
        total_time = time.time() - start_time
        generated_text = tok.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        adaptive_ratio = adaptive_decisions / max(total_decisions, 1)
        return generated_text, (output_ids.shape[1] - input_ids.shape[1]), total_time, adaptive_ratio
    
    def extract_code(text: str) -> str:
        """Extract Python code from model output."""
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        code = blocks[-1].strip() if blocks else text.strip()
        code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```\s*$", "", code)
        return code
    
    def _run_test_with_timeout(module_src: str, entry_point: str, timeout_seconds: int = 10):
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
    
    def evaluate_completion(prompt_src: str, test_src: str, entry_point: str, code: str, timeout_s: int = 10) -> bool:
        """Evaluate if generated code passes HumanEval tests."""
        if not code:
            return False
        module_src = prompt_src + "\n" + code + "\n\n" + test_src
        ok, _ = _run_test_with_timeout(module_src, entry_point, timeout_seconds=timeout_s)
        return ok
    
    def load_model(model_id: str, hf_token: str | None):
        """Load model and tokenizer."""
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, token=hf_token, torch_dtype=dtype, device_map="auto"
            )
            model.eval()
            return tok, model
        except GatedRepoError as e:
            print(f"[SKIP] Gated repo (no access): {model_id}\n  {e}")
            return None, None
        except OSError as e:
            msg = str(e)
            if "gated repo" in msg.lower() or "401" in msg or "403" in msg or "unauthorized" in msg.lower():
                print(f"[SKIP] Unauthorized / gated model: {model_id}\n  {e}")
                return None, None
            raise

# ============================================================
# Uncertainty Estimation
# ============================================================

@torch.inference_mode()
def estimate_uncertainty(
    tok,
    model,
    prompt: str,
    generated_code: str,
    sep_probe: Optional[Tuple] = None,
    layers: List[int] = [-3, -2, -1],
) -> float:
    """
    Estimate uncertainty using SEP probe.
    
    Returns:
        Probability of high semantic entropy (0-1), higher = more uncertain
    """
    if sep_probe is None:
        return 0.5  # Default uncertainty if probe not available
    
    scaler, clf, threshold = sep_probe
    
    # Reconstruct full generation context to extract features
    user_prompt = prompt + "\n\n# Your code below:\n"
    chat_text = build_chat_text(tok, user_prompt)
    
    # Get the full sequence (prompt + generated code)
    full_text = chat_text + generated_code
    input_ids = tok(full_text, return_tensors="pt").input_ids.to(model.device)
    
    # Extract SLT features from the end of the sequence
    full_ids_cpu = input_ids.detach().cpu()
    feat = extract_slt_vec_multi_layer(tok, model, full_ids_cpu, layers=layers)
    
    # Predict uncertainty using SEP probe
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    prob_high_entropy = clf.predict_proba(feat_scaled)[0, 1]  # Probability of high semantic entropy
    
    return float(prob_high_entropy)

# ============================================================
# Correction Strategies
# ============================================================

@torch.inference_mode()
def resample_code(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_samples: int = 3,
) -> List[Tuple[str, float]]:
    """
    Generate multiple resamples and return them with their uncertainties.
    
    Returns:
        List of (code, uncertainty) tuples
    """
    user_prompt = prompt + "\n\n# Your code below:\n"
    chat_text = build_chat_text(tok, user_prompt)
    
    samples = []
    for _ in range(num_samples):
        enc = tok(chat_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)
        code = extract_code(gen_text)
        samples.append(code)
    
    return samples

# ============================================================
# Main Self-Correction Function
# ============================================================

def correct_code(
    tok,
    model,
    prompt: str,
    test_src: str,
    entry_point: str,
    sep_probe: Optional[Tuple] = None,
    cfg: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generate code with automatic self-correction.
    
    Args:
        tok: Tokenizer
        model: Language model
        prompt: The code generation prompt
        test_src: Test code for evaluation
        entry_point: Function entry point name
        sep_probe: Trained SEP probe (scaler, clf, threshold)
        cfg: Configuration dictionary
    
    Returns:
        Dictionary with:
        - final_code: Final generated code
        - uncertainty_score: Final uncertainty score
        - num_corrections: Number of corrections made
        - correction_history: List of correction attempts
        - is_correct: Whether final code passes tests
        - total_time: Total generation time
    """
    if cfg is None:
        cfg = get_config()
    
    # Auto-load threshold from probe if not set in config
    if cfg["uncertainty_threshold"] is None and sep_probe is not None:
        _, _, threshold = sep_probe
        print(f"  [INFO] Using probe's recommended threshold: {threshold:.4f}")
    else:
        threshold = cfg.get("uncertainty_threshold", 0.5)
    
    max_attempts = cfg["max_attempts"]
    strategy = cfg["correction_strategy"]
    layers = cfg.get("layers", [-3, -2, -1])
    
    start_time = time.time()
    correction_history = []
    
    # Initial generation
    user_prompt = prompt + "\n\n# Your code below:\n"
    current_code, _, _ = greedy_decode(
        tok, model, user_prompt, max_new_tokens=cfg["max_new_tokens"]
    )
    current_code = extract_code(current_code)
    
    # Estimate initial uncertainty
    initial_uncertainty = estimate_uncertainty(
        tok, model, prompt, current_code, sep_probe, layers=layers
    )
    
    # DEBUG: Show initial uncertainty
    print(f"  [DEBUG] Initial uncertainty: {initial_uncertainty:.4f}, threshold: {threshold:.4f}")
    
    correction_history.append({
        "attempt": 0,
        "strategy": "initial",
        "code": current_code,
        "uncertainty": initial_uncertainty,
        "is_correct": None,  # Will evaluate later
    })
    
    best_code = current_code
    best_uncertainty = initial_uncertainty
    best_correct = None
    
    # Correction loop
    for attempt in range(1, max_attempts + 1):
        # EARLY STOP: If we found a correct solution, stop immediately
        if best_correct is True:
            print(f"  [DEBUG] Early stopping: found correct solution at attempt {attempt-1}")
            break
        
        # Check if we should correct
        if best_uncertainty <= threshold:
            print(f"  [DEBUG] Stopping corrections: uncertainty {best_uncertainty:.4f} <= threshold {threshold:.4f}")
            break  # Confident enough, stop correcting
        
        print(f"  [DEBUG] Attempt {attempt}: uncertainty {best_uncertainty:.4f} > threshold {threshold:.4f}, triggering correction...")
        
        # Apply correction strategy
        if strategy == "adaptive":
            # Use adaptive decoding
            ada_text, _, _, _ = adaptive_decode(
                tok, model, user_prompt,
                max_new_tokens=cfg["max_new_tokens"],
                beam_size=cfg["beam_size"],
                lookahead_length=cfg["lookahead_length"],
                sep_probe=sep_probe,
                token_entropy_threshold=3.5,
            )
            corrected_code = extract_code(ada_text)
            strategy_used = "adaptive"
            
        elif strategy == "resample":
            # Resample multiple times and pick best
            resamples = resample_code(
                tok, model, prompt,
                max_new_tokens=cfg["max_new_tokens"],
                temperature=cfg["resample_temperature"],
                top_p=cfg["resample_top_p"],
                num_samples=cfg["num_resamples"],
            )
            
            # Evaluate uncertainty for each resample
            resample_uncertainties = []
            for code in resamples:
                unc = estimate_uncertainty(tok, model, prompt, code, sep_probe, layers=layers)
                resample_uncertainties.append((code, unc))
            
            # Pick the one with lowest uncertainty
            resample_uncertainties.sort(key=lambda x: x[1])
            corrected_code, _ = resample_uncertainties[0]
            strategy_used = "resample"
            
        elif strategy == "both":
            # Try both strategies and pick best
            # Adaptive
            ada_text, _, _, _ = adaptive_decode(
                tok, model, user_prompt,
                max_new_tokens=cfg["max_new_tokens"],
                beam_size=cfg["beam_size"],
                lookahead_length=cfg["lookahead_length"],
                sep_probe=sep_probe,
                token_entropy_threshold=3.5,
            )
            ada_code = extract_code(ada_text)
            ada_unc = estimate_uncertainty(tok, model, prompt, ada_code, sep_probe, layers=layers)
            
            # Resample
            resamples = resample_code(
                tok, model, prompt,
                max_new_tokens=cfg["max_new_tokens"],
                temperature=cfg["resample_temperature"],
                top_p=cfg["resample_top_p"],
                num_samples=cfg["num_resamples"],
            )
            resample_uncertainties = []
            for code in resamples:
                unc = estimate_uncertainty(tok, model, prompt, code, sep_probe, layers=layers)
                resample_uncertainties.append((code, unc))
            
            # Pick best (lowest uncertainty)
            candidates = [(ada_code, ada_unc)] + resample_uncertainties
            candidates.sort(key=lambda x: x[1])
            corrected_code, _ = candidates[0]
            strategy_used = "both"
            
        else:
            # Default: just resample
            resamples = resample_code(
                tok, model, prompt,
                max_new_tokens=cfg["max_new_tokens"],
                temperature=cfg["resample_temperature"],
                top_p=cfg["resample_top_p"],
                num_samples=1,
            )
            corrected_code = resamples[0] if resamples else best_code
            strategy_used = "resample"
        
        # Estimate uncertainty for corrected code
        corrected_uncertainty = estimate_uncertainty(
            tok, model, prompt, corrected_code, sep_probe, layers=layers
        )
        
        print(f"  [DEBUG] After correction: uncertainty {corrected_uncertainty:.4f} (was {best_uncertainty:.4f})")
        
        # Evaluate correctness
        corrected_correct = evaluate_completion(
            prompt, test_src, entry_point, corrected_code, timeout_s=cfg["test_timeout_s"]
        )
        
        print(f"  [DEBUG] Correction result: correct={corrected_correct}, uncertainty_reduction={best_uncertainty - corrected_uncertainty:.4f}")
        
        correction_history.append({
            "attempt": attempt,
            "strategy": strategy_used,
            "code": corrected_code,
            "uncertainty": corrected_uncertainty,
            "is_correct": corrected_correct,
        })
        
        # Update best if this is better (prioritize correctness, then lower uncertainty)
        if corrected_correct and (best_correct is None or not best_correct):
            best_code = corrected_code
            best_uncertainty = corrected_uncertainty
            best_correct = corrected_correct
        elif corrected_uncertainty < best_uncertainty and (best_correct is None or not best_correct):
            best_code = corrected_code
            best_uncertainty = corrected_uncertainty
            best_correct = corrected_correct
        
        # If we found a correct solution, we can stop early
        if best_correct:
            break
    
    # Final evaluation
    if best_correct is None:
        best_correct = evaluate_completion(
            prompt, test_src, entry_point, best_code, timeout_s=cfg["test_timeout_s"]
        )
    
    total_time = time.time() - start_time
    
    # DEBUG: Summary
    uncertainty_reduction = initial_uncertainty - best_uncertainty
    print(f"  [DEBUG Summary] Initial: {initial_uncertainty:.4f}, Final: {best_uncertainty:.4f}, Reduction: {uncertainty_reduction:.4f}")
    print(f"  [DEBUG Summary] Corrections made: {len(correction_history) - 1}, Final correct: {best_correct}")
    
    if len(correction_history) == 1 and initial_uncertainty > threshold:
        print(f"  [WARNING] No corrections made despite high initial uncertainty ({initial_uncertainty:.4f} > {threshold:.4f})")
        print(f"            This may indicate an issue with correction triggering logic")
    
    return {
        "final_code": best_code,
        "uncertainty_score": best_uncertainty,
        "initial_uncertainty": initial_uncertainty,
        "num_corrections": len(correction_history) - 1,
        "correction_history": correction_history,
        "is_correct": best_correct,
        "total_time": total_time,
    }

# ============================================================
# Evaluation Function
# ============================================================

def evaluate_self_correction(
    tasks: List[Dict[str, Any]],
    tok,
    model,
    cfg,
    sep_probe: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """Evaluate self-correction on HumanEval tasks."""
    
    n = len(tasks)
    if n == 0:
        return {"error": "No tasks provided"}
    
    results = {
        "baseline_pass": 0,
        "corrected_pass": 0,
        "baseline_latencies": [],
        "corrected_latencies": [],
        "uncertainty_reductions": [],
        "num_corrections": [],
        "task_results": [],
    }
    
    for task in tqdm(tasks, desc="Evaluating self-correction", unit="task"):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_src = task["test"]
        entry_point = task["entry_point"]
        
        # Baseline: greedy decode
        user_prompt = prompt + "\n\n# Your code below:\n"
        base_text, _, base_latency = greedy_decode(
            tok, model, user_prompt, max_new_tokens=cfg["max_new_tokens"]
        )
        base_code = extract_code(base_text)
        base_correct = evaluate_completion(
            prompt, test_src, entry_point, base_code, timeout_s=cfg["test_timeout_s"]
        )
        
        if base_correct:
            results["baseline_pass"] += 1
        results["baseline_latencies"].append(base_latency)
        
        # Self-correction
        correction_result = correct_code(
            tok, model, prompt, test_src, entry_point, sep_probe=sep_probe, cfg=cfg
        )
        
        if correction_result["is_correct"]:
            results["corrected_pass"] += 1
        results["corrected_latencies"].append(correction_result["total_time"])
        results["uncertainty_reductions"].append(
            correction_result["initial_uncertainty"] - correction_result["uncertainty_score"]
        )
        results["num_corrections"].append(correction_result["num_corrections"])
        
        results["task_results"].append({
            "task_id": task_id,
            "baseline_correct": base_correct,
            "corrected_correct": correction_result["is_correct"],
            "baseline_latency": base_latency,
            "corrected_latency": correction_result["total_time"],
            "initial_uncertainty": correction_result["initial_uncertainty"],
            "final_uncertainty": correction_result["uncertainty_score"],
            "uncertainty_reduction": correction_result["initial_uncertainty"] - correction_result["uncertainty_score"],
            "num_corrections": correction_result["num_corrections"],
            "improved": correction_result["is_correct"] and not base_correct,
            "degraded": base_correct and not correction_result["is_correct"],
        })
    
    # Compute summary statistics
    results["baseline_pass_at_1"] = results["baseline_pass"] / n
    results["corrected_pass_at_1"] = results["corrected_pass"] / n
    results["improvement"] = results["corrected_pass_at_1"] - results["baseline_pass_at_1"]
    results["avg_baseline_latency"] = float(np.mean(results["baseline_latencies"]))
    results["avg_corrected_latency"] = float(np.mean(results["corrected_latencies"]))
    results["avg_uncertainty_reduction"] = float(np.mean(results["uncertainty_reductions"]))
    results["avg_num_corrections"] = float(np.mean(results["num_corrections"]))
    
    # Count improvements/degradations
    results["num_improved"] = sum(1 for r in results["task_results"] if r["improved"])
    results["num_degraded"] = sum(1 for r in results["task_results"] if r["degraded"])
    
    # DEBUG: Analyze correction behavior
    avg_corrections = results["avg_num_corrections"]
    avg_uncertainty_reduction = results["avg_uncertainty_reduction"]
    
    if avg_corrections < 0.1:
        print(f"\n⚠️  WARNING: Self-correction rarely triggered (avg corrections: {avg_corrections:.2f})")
        print(f"   This suggests the uncertainty threshold ({cfg.get('uncertainty_threshold', 0.5)}) may be too high")
        print(f"   Check probe training output for recommended threshold")
    
    if abs(avg_uncertainty_reduction) < 0.01:
        print(f"\n⚠️  WARNING: Minimal uncertainty reduction ({avg_uncertainty_reduction:.4f})")
        print(f"   This may indicate:")
        print(f"   - Corrections not being applied effectively")
        print(f"   - Probe predictions not changing after corrections")
        print(f"   - Threshold too high (corrections never trigger)")
    
    return results

# ============================================================
# Main
# ============================================================

def main():
    """Demo of self-correction on HumanEval problems."""
    print("=== Self-Correction with SEP Probe ===")
    cfg = get_config()
    
    # HF token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with Llama access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    
    # Load dataset
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    tasks = [ds[i] for i in range(len(ds))]
    
    if cfg["limit_tasks"] is not None:
        tasks = tasks[:cfg["limit_tasks"]]
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
        probe_dir = cfg["probe_path"]
        scaler, clf, threshold = load_sep_probe(probe_dir)
        if scaler is not None:
            sep_probe = (scaler, clf, threshold)
            print(f"✅ Using SEP probe from {probe_dir}")
        else:
            print(f"⚠️  SEP probe not found, self-correction will use fallback")
    else:
        print("⚠️  SEP probe disabled, self-correction may not work optimally")
    
    # Evaluate
    print("\n" + "="*80)
    print("Starting self-correction evaluation...")
    print(f"Uncertainty threshold: {cfg['uncertainty_threshold']}")
    print(f"Max attempts: {cfg['max_attempts']}")
    print(f"Correction strategy: {cfg['correction_strategy']}")
    print("="*80)
    
    results = evaluate_self_correction(tasks, tok, model, cfg, sep_probe=sep_probe)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Baseline (Greedy) Pass@1: {results['baseline_pass_at_1']:.4f} ({results['baseline_pass']}/{len(tasks)})")
    print(f"Self-Corrected Pass@1:    {results['corrected_pass_at_1']:.4f} ({results['corrected_pass']}/{len(tasks)})")
    print(f"Improvement:              {results['improvement']:+.4f}")
    print(f"\nTasks improved:  {results['num_improved']}")
    print(f"Tasks degraded:  {results['num_degraded']}")
    print(f"\nAvg baseline latency: {results['avg_baseline_latency']:.3f}s")
    print(f"Avg corrected latency:  {results['avg_corrected_latency']:.3f}s")
    print(f"Avg uncertainty reduction: {results['avg_uncertainty_reduction']:.4f}")
    print(f"Avg corrections per task: {results['avg_num_corrections']:.2f}")
    
    # Save results
    output_file = "self_correction_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()

