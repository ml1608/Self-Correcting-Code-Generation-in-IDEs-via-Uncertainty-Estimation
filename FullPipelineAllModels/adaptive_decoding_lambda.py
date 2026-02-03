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
        
        # Adaptive decoding parameters
        "max_new_tokens": 256,
        "beam_size": 3,
        "lookahead_length": 5,
        
        # SEP probe settings
        "probe_path": "sep_slt_runs/meta-llama_Llama-3.2-3B-Instruct",  # Path to trained probe
        "use_sep_probe": True,  # Use SEP probe to trigger adaptive decoding
        "fallback_token_entropy_threshold": 3.5,  # Fallback if probe not available
        
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
# Load SEP Probe
# ============================================================

def load_sep_probe(probe_dir: str, threshold_override: Optional[float] = None, feature_method: str = "SLT"):
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
            print(f"✅ Loaded SEP probe: using recommended threshold={threshold:.4f} (from probe training)")
        elif "threshold" in probe_data:
            threshold = probe_data["threshold"]
            print(f"✅ Loaded SEP probe: using semantic entropy threshold={threshold:.4f} (fallback)")
        else:
            threshold = 0.5  # Default fallback
            print(f"⚠️  No threshold found in probe, using default={threshold:.4f}")
        
        print(f"✅ Loaded SEP probe from pickle: {classifier_type}, feature_method={feature_method}, threshold={threshold:.4f}")
        return scaler, clf, threshold, feature_method
    
    # Fallback: try to reconstruct from JSON (limited support)
    if os.path.exists(probe_json_path):
        print(f"⚠️  Pickle file not found, attempting to load from JSON (limited support)")
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
        print(f"⚠️  Cannot fully reconstruct {classifier_type} from JSON. Please use probe.pkl")
        return None, None, None, feature_method
    
    print(f"⚠️  Probe not found at {probe_dir}")
    return None, None, None, feature_method

# ============================================================
# Feature Extraction (SLT and TBG multi-layer)
# ============================================================

@torch.inference_mode()
def extract_features_multi_method(tok, model, full_ids_cpu: torch.Tensor, prompt_len: int, 
                                   layers: list = [-3, -2, -1], method: str = "SLT"):
    """
    Extract features using different methods:
    - SLT: second-to-last token (index -2)
    - TBG: token before generation (prompt_len - 1)
    """
    # Ensure proper shape: [batch_size, seq_len]
    if full_ids_cpu.dim() == 1:
        full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
    else:
        full_ids = full_ids_cpu.to(model.device)
    out = model(full_ids, output_hidden_states=True, use_cache=False)
    
    # Extract features from multiple layers
    features = []
    for layer in layers:
        hs = out.hidden_states[layer]
        
        if method == "SLT":
            token_idx = -2
        elif method == "TBG":
            token_idx = prompt_len - 1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle edge cases
        if token_idx < 0:
            token_idx = hs.shape[1] + token_idx
        
        if token_idx >= hs.shape[1]:
            token_idx = hs.shape[1] - 1
        
        if token_idx < 0:
            token_idx = 0
        
        features.append(hs[0, token_idx, :].float().detach().cpu().numpy())
    
    # Concatenate all layer features
    return np.concatenate(features)

@torch.inference_mode()
def extract_slt_vec_multi_layer(tok, model, full_ids_cpu: torch.Tensor, layers: list = [-3, -2, -1]):
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

def build_chat_text(tok, user_prompt: str):
    """Build chat-formatted text for Llama."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tok, "chat_template", None) not in (None, ""):
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"[SYSTEM] {SYSTEM_PROMPT}\n[USER] {user_prompt}\n[ASSISTANT]\n"

# ============================================================
# Decoding Functions
# ============================================================

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
def adaptive_decode(
    tok, 
    model, 
    prompt: str, 
    max_new_tokens: int = 256,
    beam_size: int = 3,
    lookahead_length: int = 5,
    sep_probe: Optional[Tuple] = None,  # (scaler, clf, threshold, feature_method)
    token_entropy_threshold: float = 3.5,
) -> Tuple[str, int, float]:
    """
    Adaptive decoding: uses SEP probe to predict semantic entropy.
    If predicted high semantic entropy, uses beam search with lookahead.
    Otherwise, uses greedy decoding.
    """
    chat_text = build_chat_text(tok, prompt)
    input_ids = tok(chat_text, return_tensors="pt").input_ids.to(model.device)
    output_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]  # Store prompt length for TBG feature extraction
    start_time = time.time()
    
    # Track adaptive decisions
    adaptive_decisions = 0
    total_decisions = 0

    for step in range(max_new_tokens):
        outputs = model(output_ids, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Check if we should use adaptive decoding
        use_adaptive = False
        
        if sep_probe is not None:
            # Use SEP probe to predict semantic entropy
            # sep_probe can be (scaler, clf, threshold) or (scaler, clf, threshold, feature_method)
            if len(sep_probe) == 4:
                scaler, clf, threshold, feature_method = sep_probe
            else:
                scaler, clf, threshold = sep_probe
                feature_method = "SLT"  # Default to SLT for backward compatibility
            
            # Extract features using the specified method
            full_ids_cpu = output_ids.detach().cpu()
            feat = extract_features_multi_method(
                tok, model, full_ids_cpu, prompt_len, 
                layers=[-3, -2, -1], method=feature_method
            )
            
            # Predict high semantic entropy (y=1)
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            prob_high_entropy = clf.predict_proba(feat_scaled)[0, 1]  # Probability of high semantic entropy
            
            # DEBUG: Track predictions (sample first 3 steps and every 10th step to avoid spam)
            if step < 3 or step % 10 == 0:
                print(f"    [Step {step}] prob={prob_high_entropy:.4f}, threshold={threshold:.4f}, adaptive={prob_high_entropy > threshold}")
            
            # Use adaptive if predicted high semantic entropy (using auto-loaded threshold)
            use_adaptive = prob_high_entropy > threshold 
        else:
            # Fallback: use token entropy
            p = probs[0].detach().float().cpu().numpy()
            entropy = -float(np.sum(p * np.log(p + 1e-10)))
            use_adaptive = entropy > token_entropy_threshold
        
        total_decisions += 1
        
        if use_adaptive:
            adaptive_decisions += 1
            # Beam search with lookahead
            topk = torch.topk(probs, beam_size, dim=-1)
            candidate_tokens = topk.indices[0]
            candidate_probs = topk.values[0]

            best_score = -float("inf")
            best_token = None

            for token, token_prob in zip(candidate_tokens, candidate_probs):
                sim_ids = torch.cat([output_ids, token.view(1, 1)], dim=1)
                score = math.log(float(token_prob) + 1e-10)

                # Lookahead
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
            # Greedy decoding
            next_token_id = int(torch.argmax(probs))

        output_ids = torch.cat([output_ids, torch.tensor([[next_token_id]], device=model.device)], dim=1)

        if next_token_id == tok.eos_token_id:
            break

    total_time = time.time() - start_time
    generated_text = tok.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    
    adaptive_ratio = adaptive_decisions / max(total_decisions, 1)
    
    # DEBUG: Warning if adaptive never triggered (only print once per generation, not for every step)
    if sep_probe is not None and adaptive_ratio == 0.0 and total_decisions > 5:
        pass  # Warning will be shown at task level instead
    
    return generated_text, (output_ids.shape[1] - input_ids.shape[1]), total_time, adaptive_ratio

# ============================================================
# Code Extraction and Evaluation
# ============================================================

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

# ============================================================
# Main Evaluation
# ============================================================

def evaluate_adaptive_decoding(
    tasks: List[Dict[str, Any]],
    tok,
    model,
    cfg,
    sep_probe: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """Evaluate adaptive decoding vs greedy baseline on HumanEval."""
    
    n = len(tasks)
    if n == 0:
        return {"error": "No tasks provided"}
    
    results = {
        "baseline_pass": 0,
        "adaptive_pass": 0,
        "baseline_latencies": [],
        "adaptive_latencies": [],
        "adaptive_ratios": [],  # Fraction of steps that used adaptive decoding
        "task_results": [],
    }
    
    # Track probe predictions for analysis
    all_probe_predictions = []
    
    for task in tqdm(tasks, desc="Evaluating adaptive decoding", unit="task"):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_src = task["test"]
        entry_point = task["entry_point"]
        
        user_prompt = prompt + "\n\n# Your code below:\n"
        
        # Baseline: greedy decode
        base_text, base_len, base_latency = greedy_decode(tok, model, user_prompt, max_new_tokens=cfg["max_new_tokens"])
        base_code = extract_code(base_text)
        base_correct = evaluate_completion(prompt, test_src, entry_point, base_code, timeout_s=cfg["test_timeout_s"])
        
        if base_correct:
            results["baseline_pass"] += 1
        results["baseline_latencies"].append(base_latency)
        
        # Adaptive decode
        ada_text, ada_len, ada_latency, ada_ratio = adaptive_decode(
            tok, model, user_prompt,
            max_new_tokens=cfg["max_new_tokens"],
            beam_size=cfg["beam_size"],
            lookahead_length=cfg["lookahead_length"],
            sep_probe=sep_probe,
            token_entropy_threshold=cfg["fallback_token_entropy_threshold"],
        )
        ada_code = extract_code(ada_text)
        ada_correct = evaluate_completion(prompt, test_src, entry_point, ada_code, timeout_s=cfg["test_timeout_s"])
        
        if ada_correct:
            results["adaptive_pass"] += 1
        results["adaptive_latencies"].append(ada_latency)
        results["adaptive_ratios"].append(ada_ratio)
        
        # DEBUG: Warning if adaptive never triggered for this task
        if ada_ratio == 0.0 and sep_probe is not None:
            if len(results["task_results"]) < 3:  # Only show for first few tasks
                print(f"  [WARNING Task {task_id}] Adaptive decoding never triggered (ratio=0.0)")
                print(f"            Probe predictions may all be below threshold")
        
        results["task_results"].append({
            "task_id": task_id,
            "baseline_correct": base_correct,
            "adaptive_correct": ada_correct,
            "baseline_latency": base_latency,
            "adaptive_latency": ada_latency,
            "adaptive_ratio": ada_ratio,
            "improved": ada_correct and not base_correct,
            "degraded": base_correct and not ada_correct,
        })
    
    # Compute summary statistics
    results["baseline_pass_at_1"] = results["baseline_pass"] / n
    results["adaptive_pass_at_1"] = results["adaptive_pass"] / n
    results["improvement"] = results["adaptive_pass_at_1"] - results["baseline_pass_at_1"]
    results["avg_baseline_latency"] = float(np.mean(results["baseline_latencies"]))
    results["avg_adaptive_latency"] = float(np.mean(results["adaptive_latencies"]))
    results["avg_adaptive_ratio"] = float(np.mean(results["adaptive_ratios"]))
    
    # Count improvements/degradations
    results["num_improved"] = sum(1 for r in results["task_results"] if r["improved"])
    results["num_degraded"] = sum(1 for r in results["task_results"] if r["degraded"])
    
    # DEBUG: Analyze adaptive ratios
    avg_adaptive_ratio = results["avg_adaptive_ratio"]
    if avg_adaptive_ratio < 0.01:
        print(f"\n⚠️  WARNING: Adaptive decoding rarely triggered (avg ratio: {avg_adaptive_ratio:.2%})")
        print(f"   This suggests the threshold may be too high")
        print(f"   Check probe training output for recommended threshold")
    elif avg_adaptive_ratio > 0.9:
        print(f"\n⚠️  WARNING: Adaptive decoding triggered too often (avg ratio: {avg_adaptive_ratio:.2%})")
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
        if "gated repo" in msg.lower() or "401" in msg or "403" in msg or "unauthorized" in msg.lower():
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
            print(f"⚠️  SEP probe not found, using token entropy fallback")
    else:
        print("Using token entropy threshold (SEP probe disabled)")
    
    # Evaluate
    print("\n" + "="*80)
    print("Starting evaluation...")
    print(f"Beam size: {cfg['beam_size']}")
    print(f"Lookahead length: {cfg['lookahead_length']}")
    print("="*80)
    
    results = evaluate_adaptive_decoding(tasks, tok, model, cfg, sep_probe=sep_probe)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Baseline (Greedy) Pass@1: {results['baseline_pass_at_1']:.4f} ({results['baseline_pass']}/{len(tasks)})")
    print(f"Adaptive Pass@1:         {results['adaptive_pass_at_1']:.4f} ({results['adaptive_pass']}/{len(tasks)})")
    print(f"Improvement:             {results['improvement']:+.4f}")
    print(f"\nTasks improved:  {results['num_improved']}")
    print(f"Tasks degraded:  {results['num_degraded']}")
    print(f"\nAvg baseline latency: {results['avg_baseline_latency']:.3f}s")
    print(f"Avg adaptive latency:  {results['avg_adaptive_latency']:.3f}s")
    print(f"Avg adaptive ratio:    {results['avg_adaptive_ratio']:.2%} (fraction of steps using adaptive)")
    
    # Save results
    output_file = "adaptive_decoding_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()

