#!/usr/bin/env python3
"""
Train SEP Probes (SLT and TBG features with entropy regression probes)

This script:
1. Samples code ONCE (M_samples=20) for all methods
2. Extracts features using SLT and TBG methods
3. Trains regression probes to predict semantic entropy directly
4. Saves probes and dataset splits

Probe options:
- LINREG: Linear regression probe (predicts semantic entropy directly)
"""

import os
import re
import io
import gc
import json
import time
import random
import hashlib
import signal
import pickle
import sys
import tempfile
import subprocess
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# Configuration
# ============================================================

def get_experiment_config():
    return {
        # Models to train probes for (from MTEmodels notebook)
        "models": [
            ("llama", "3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
            ("qwen-coder-instruct", "3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
            ("deepseek", "3B-Instruct", "deepseek-ai/deepseek-coder-1.3b-instruct"),
        ],
        
        # Dataset configuration - supports HumanEval and BigCodeBench
        # For HumanEval: dataset_name="openai_humaneval", split="test"
        # For BigCodeBench: dataset_name="bigcode/bigcodebench", split="v0.1.4"
        "dataset_name": "openai_humaneval",
        "split": "test",
        # For symbolic execution clustering, this must be executable Python context.
        # Use complete_prompt (not instruct_prompt) on BigCodeBench.
        "prompt_field": "prompt",  # BigCodeBench: prefer "complete_prompt"
        "limit_tasks": None,  # None = all tasks (1140 for BigCodeBench, 164 for HumanEval)
        
        # Sampling (done ONCE for all methods)
        "M_samples": 5,
        
        "sample_max_new_tokens": 256,
        "sample_temperature": 0.7,
        "sample_top_p": 0.95,
        "greedy_max_new_tokens": 256,
        "test_timeout_s": 10,
        "parallel_workers": 5,  # Number of parallel workers for code execution
        "seed": 42,
        
        # Feature extraction methods to use (only SLT and TBG)
        "feature_methods": ["SLT", "TBG"],
        
        # Layers to use
        "layers": [-3, -2, -1],
        # "layers": [-1],
        
        # Probe models to train
        # - "linreg": linear regression probe (predicts semantic entropy directly)
        "classifiers": ["linreg"],
        
        # Dataset caching
        # If True, skip building dataset if it already exists (faster for re-training probes)
        # If False, always rebuild the dataset (use when changing sampling parameters)
        "skip_existing_dataset": True,
        
        # Labeling config kept only for backward-compatible analytics/splits.
        "label_mode": "median",
        # Semantic clustering method (strict: no fallback)
        "cluster_method": "symbolic_execution",
    }


# ============================================================
# Dataset Field Mapping
# ============================================================

def get_task_fields(task: dict, dataset_name: str, prompt_field: str = "instruct_prompt") -> dict:
    """
    Map dataset fields to standard names for different datasets.
    
    Args:
        task: Raw task from the dataset
        dataset_name: Name of the dataset (e.g., "bigcode/bigcodebench", "openai_humaneval")
        prompt_field: Field to use for the prompt (for BigCodeBench: "instruct_prompt" or "complete_prompt")
    
    Returns:
        dict with standardized fields: task_id, prompt, test, entry_point, code_prompt
    """
    if "bigcodebench" in dataset_name.lower():
        # Symbolic execution requires prompt context that can be executed as Python.
        # complete_prompt contains runnable code scaffolding, while instruct_prompt is NL.
        prompt = task.get(prompt_field, "")
        if prompt_field == "instruct_prompt" and task.get("complete_prompt"):
            prompt = task["complete_prompt"]
        return {
            "task_id": task["task_id"],
            "prompt": prompt,
            "test": task["test"],
            "entry_point": task["entry_point"],  # Always "task_func" for BigCodeBench
            "code_prompt": task.get("code_prompt", ""),
            "canonical_solution": task.get("canonical_solution", ""),
        }
    else:  # HumanEval or similar
        return {
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "test": task["test"],
            "entry_point": task["entry_point"],
            "code_prompt": "",
            "canonical_solution": task.get("canonical_solution", ""),
        }


def parse_array_string(arr_str):
    """
    Parse a string representation of a numpy array back into a numpy array.
    Handles formats like: '[ 5.5  0.195  -2.765  ... ]' or full arrays.
    
    Returns:
        numpy array if successful, None if array is truncated (contains '...')
    """
    import ast
    
    # If already an array or list, return as numpy array
    if isinstance(arr_str, np.ndarray):
        return arr_str.astype(np.float32)
    if isinstance(arr_str, list):
        return np.array(arr_str, dtype=np.float32)
    
    arr_str_orig = str(arr_str).strip()
    
    # Check for truncation marker - can't recover truncated arrays
    if '...' in arr_str_orig:
        return None  # Signal that array is truncated
    
    # Method 1: Try ast.literal_eval (safest, handles full arrays as Python lists)
    try:
        arr = ast.literal_eval(arr_str_orig)
        if isinstance(arr, list):
            return np.array(arr, dtype=np.float32)
    except (ValueError, SyntaxError):
        pass
    
    # Method 2: Try parsing as space-separated numbers
    try:
        cleaned = arr_str_orig.replace('[', '').replace(']', '').replace('\n', ' ').strip()
        nums = []
        for token in cleaned.split():
            token = token.strip()
            if token:
                try:
                    nums.append(float(token))
                except ValueError:
                    pass
        if nums:
            return np.array(nums, dtype=np.float32)
    except Exception:
        pass
    
    return None  # Failed to parse


SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function so that it passes the tests. "
    "Return only Python code, no explanation."
)

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_symbolic_setup():
    """
    Strict preflight checks for symbolic-execution clustering.
    Raises RuntimeError with actionable guidance if anything is missing.
    """
    helper = os.environ.get("PYEXZ3_CLUSTER_SCRIPT", "").strip()
    if not helper:
        raise RuntimeError(
            "Missing PYEXZ3_CLUSTER_SCRIPT. "
            "Set it to the absolute path of your symbolic clustering helper."
        )
    if not os.path.exists(helper):
        raise RuntimeError(
            f"PYEXZ3_CLUSTER_SCRIPT does not exist: {helper}"
        )

    # Validate helper invocation contract directly.
    try:
        proc = subprocess.run(
            [sys.executable, helper, "--self-check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to execute symbolic helper: {e}") from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"Symbolic helper self-check failed (code={proc.returncode}): {proc.stderr.strip()}"
        )

# ============================================================
# Helper Functions
# ============================================================

def build_prompt_for_model(raw_prompt: str, family: str):
    """Build prompt based on model family (from MTEmodels notebook)."""
    if family in {"llama", "deepseek"}:
        system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
        user = raw_prompt + "\n\n# Your code below:\n"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return {"chat": True, "messages": messages}
    
    if family == "qwen-coder" or family == "qwen-coder-instruct":
        system = "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
        user = raw_prompt + "\n\n# Your code below:\n"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return {"chat": True, "messages": messages}
    
    return {"chat": False, "text": raw_prompt + "\n# Your code below:\n"}

def build_chat_text(tok, user_prompt: str, family: str = "llama"):
    """Build chat text for model, handling different model families."""
    spec = build_prompt_for_model(user_prompt, family)
    has_chat_template = getattr(tok, "chat_template", None) not in (None, "")
    
    if spec["chat"] and has_chat_template:
        return tok.apply_chat_template(
            spec["messages"], add_generation_prompt=True, tokenize=False
        )
    elif spec["chat"]:
        # Fallback: build text manually
        parts = []
        for msg in spec["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"[SYSTEM] {content}\n")
            elif role == "user":
                parts.append(f"[USER] {content}\n")
            else:
                parts.append(f"[{role.upper()}] {content}\n")
        return "".join(parts) + "\n[ASSISTANT]\n"
    else:
        return spec["text"]

def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = blocks[-1].strip() if blocks else text.strip()
    code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```\s*$", "", code)
    try:
        compile(code, "<candidate>", "exec")
        return code
    except SyntaxError:
        return ""

def _run_test_with_timeout(module_src: str, entry_point: str, timeout_seconds: int = 10):
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

def semantic_signature(prompt_src: str, test_src: str, entry_point: str, code: str, timeout_s: int = 10):
    if not code:
        return "INVALID:syntax", 0
    module_src = prompt_src + "\n" + code + "\n\n" + test_src
    ok, logs = _run_test_with_timeout(module_src, entry_point, timeout_seconds=timeout_s)
    logs_norm = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", logs)
    h = hashlib.sha256(logs_norm.encode("utf-8", errors="ignore")).hexdigest()[:16]
    sig = ("PASS" if ok else "FAIL") + ":" + h
    return sig, int(ok)


def symbolic_signature_with_pyexz3(
    prompt_src: str, test_src: str, entry_point: str, code: str, timeout_s: int = 10
):
    """
    Compute semantic cluster signature with symbolic execution using an external helper.

    Expected helper path via PYEXZ3_CLUSTER_SCRIPT.
    Helper contract:
      - argv: <helper.py> --entry-point <entry_point> --timeout <seconds> <module_file.py>
      - stdout JSON: {"cluster_id": "...", "passed": 0/1}

    Returns:
      (sig, ok) on success.
    Raises:
      RuntimeError on missing helper or execution/parsing failures.
    """
    helper = os.environ.get("PYEXZ3_CLUSTER_SCRIPT", "").strip()
    if not helper or not os.path.exists(helper):
        raise RuntimeError(
            "Symbolic execution helper not configured. "
            "Set PYEXZ3_CLUSTER_SCRIPT to a valid helper script path."
        )

    if not code:
        return "INVALID:syntax", 0

    module_src = prompt_src + "\n" + code + "\n\n" + test_src
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmpf:
            tmpf.write(module_src)
            tmp_path = tmpf.name

        cmd = [
            sys.executable,
            helper,
            "--entry-point",
            entry_point,
            "--timeout",
            str(timeout_s),
            tmp_path,
        ]
        # Parent timeout must exceed helper internal timeout + exec/import overhead.
        parent_timeout = max(timeout_s + 35, 60)
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=parent_timeout,
            )
        except subprocess.TimeoutExpired:
            h = hashlib.sha256(
                module_src.encode("utf-8", errors="ignore")
            ).hexdigest()[:16]
            return f"SYM:SUBPROC_TIMEOUT:{h}", 0
        if proc.returncode != 0:
            raise RuntimeError(
                f"Symbolic execution helper failed (code={proc.returncode}): {proc.stderr.strip()}"
            )

        raw_out = (proc.stdout or "").strip()
        if not raw_out:
            raise RuntimeError("Symbolic helper returned empty stdout.")

        # Robust parsing: helper may emit extra lines/logs.
        # Try full stdout, then last non-empty line, then python-literal fallback.
        parsed = None
        candidates = [raw_out]
        lines = [ln.strip() for ln in raw_out.splitlines() if ln.strip()]
        if lines:
            candidates.append(lines[-1])

        for cand in candidates:
            try:
                parsed = json.loads(cand)
                break
            except Exception:
                pass
            try:
                literal_obj = ast.literal_eval(cand)
                if isinstance(literal_obj, dict):
                    parsed = literal_obj
                    break
            except Exception:
                pass

        if parsed is None:
            raise RuntimeError(f"Unable to parse helper output as JSON/dict: {raw_out[:400]}")
        data = parsed
        cluster_id = str(data.get("cluster_id", "")).strip()
        if not cluster_id:
            raise RuntimeError("Symbolic execution helper returned empty cluster_id.")
        ok = int(bool(data.get("passed", 0)))
        sig = f"SYM:{cluster_id}"
        return sig, ok
    except Exception as e:
        raise RuntimeError(f"Symbolic execution clustering failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def parallel_semantic_signatures(prompt_src: str, test_src: str, entry_point: str, 
                                 codes: list, timeout_s: int = 10, max_workers: int = 4,
                                 cluster_method: str = "trace_hash"):
    """
    Run semantic signature checks in parallel using ThreadPoolExecutor.
    
    This is faster than sequential execution because code execution is I/O bound
    (subprocess calls) and can be parallelized across CPU cores.
    
    Args:
        prompt_src: The original prompt source code
        test_src: The test source code
        entry_point: Function entry point for testing
        codes: List of generated code strings to evaluate
        timeout_s: Timeout per code execution
        max_workers: Number of parallel workers (default 4)
    
    Returns:
        List of (signature, pass_flag) tuples in the same order as input codes
    """
    def evaluate_one(code):
        if cluster_method == "symbolic_execution":
            return symbolic_signature_with_pyexz3(
                prompt_src, test_src, entry_point, code, timeout_s
            )
        return semantic_signature(prompt_src, test_src, entry_point, code, timeout_s)
    
    # Use ThreadPoolExecutor for parallel I/O-bound execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(evaluate_one, codes))
    
    return results


def softmax_probs_from_len_norm_logps(logps):
    a = np.array(logps, dtype=np.float64)
    a = a - np.max(a)
    p = np.exp(a)
    denom = np.sum(p)
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(p) / max(len(p), 1)
    return p / denom

def entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))

# ============================================================
# Model Loading
# ============================================================

def load_model(model_id: str, hf_token: str | None):
    # Debug CUDA availability
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  ⚠️  CUDA not available! Install PyTorch with CUDA support:")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        # Try to use Flash Attention 2 for faster generation (2-3x speedup)
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
            if attn_impl:
                print(f"  ✅ Using Flash Attention 2")
        except Exception as e:
            # Fall back to default attention if Flash Attention not available
            print(f"  ⚠️  Flash Attention 2 not available ({e}), using default")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                torch_dtype=dtype,
            )
        
        # Explicitly move model to GPU
        model = model.to(device)
        model.eval()
        print(f"  Model device: {device}")
        
        # Reset model's generation_config to avoid conflicts with our sampling settings
        # Some models (e.g., Qwen) have custom configs that override do_sample, top_k, etc.
        if hasattr(model, 'generation_config'):
            model.generation_config.do_sample = None
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        
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
# Generation Functions
# ============================================================

@torch.inference_mode()
def sample_completion_with_len_norm_logp(tok, model, chat_text: str, max_new_tokens: int, temperature: float, top_p: float):
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full_ids = out.sequences[0]
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = full_ids[prompt_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    
    logps = []
    for t, step_logits in enumerate(out.scores):
        if t >= len(gen_ids):
            break
        token_id = gen_ids[t].item()
        lprobs = torch.log_softmax(step_logits[0], dim=-1)
        logps.append(lprobs[token_id].item())
    
    L = max(len(logps), 1)
    len_norm_logp = float(np.sum(logps) / L)
    return {"text": gen_text, "len_norm_logp": len_norm_logp}


@torch.inference_mode()
def batch_sample_completions(tok, model, chat_text: str, num_samples: int,
                             max_new_tokens: int, temperature: float, top_p: float):
    """
    Generate multiple samples in a single batch call for GPU parallelism.
    
    This is much faster than calling sample_completion_with_len_norm_logp() in a loop
    because it leverages GPU parallelism via num_return_sequences.
    
    Args:
        tok: Tokenizer
        model: Language model
        chat_text: The chat-formatted prompt
        num_samples: Number of samples to generate (M_samples)
        max_new_tokens: Maximum tokens per sample
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        List of dicts with "text" and "len_norm_logp" for each sample
    """
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    
    # Generate all samples in one call
    # Note: Some models (e.g., Qwen) have custom generation_config that may override settings.
    # We explicitly set top_k=None to avoid conflicts with model defaults.
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=None,  # Disable top_k to avoid conflict with model's default config
        num_return_sequences=num_samples,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    
    # out.sequences shape: [num_samples, seq_len]
    # out.scores: tuple of [num_samples, vocab_size] for each generated position
    results = []
    
    for seq_idx in range(num_samples):
        full_ids = out.sequences[seq_idx]
        gen_ids = full_ids[prompt_len:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        
        # Compute length-normalized log probability for this sample
        logps = []
        for t, step_logits in enumerate(out.scores):
            if t >= len(gen_ids):
                break
            token_id = gen_ids[t].item()
            # step_logits shape: [num_samples, vocab_size]
            lprobs = torch.log_softmax(step_logits[seq_idx], dim=-1)
            logps.append(lprobs[token_id].item())
        
        L = max(len(logps), 1)
        len_norm_logp = float(np.sum(logps) / L)
        
        results.append({"text": gen_text, "len_norm_logp": len_norm_logp})
    
    return results


@torch.inference_mode()
def greedy_generate_full_ids_and_prompt_len(tok, model, chat_text: str, max_new_tokens: int):
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full_ids = out.sequences[0].detach().cpu()
    prompt_len = int(enc["input_ids"].shape[1])
    return full_ids, prompt_len

# ============================================================
# Feature Extraction
# ============================================================

@torch.inference_mode()
def extract_features_multi_method(tok, model, full_ids_cpu: torch.Tensor, prompt_len: int, 
                                   layers: list, method: str):
    """
    Extract features using different methods:
    - SLT: second-to-last token (index -2)
    - TBG: token before generation (prompt_len - 1)
    
    NOTE: For TBG, this function is kept for backwards compatibility.
    Use extract_tbg_features_one_token() for the new 1-token TBG extraction.
    """
    full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
    out = model(full_ids, output_hidden_states=True, use_cache=False)
    
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
    
    return np.concatenate(features)


@torch.inference_mode()
def extract_tbg_features_one_token(tok, model, chat_text: str, layers: list):
    """
    Extract TBG features after generating exactly 1 token.
    
    This is the new approach for TBG feature extraction:
    1. Encode the prompt
    2. Generate exactly 1 token greedily
    3. Run forward pass with prompt + 1 token
    4. Extract features at position (prompt_len - 1)
    
    This gives the model a chance to "commit" to a direction while being
    fast enough to enable early uncertainty detection.
    
    Args:
        tok: Tokenizer
        model: Language model
        chat_text: The full chat-formatted prompt text
        layers: List of layer indices to extract features from (e.g., [-3, -2, -1])
    
    Returns:
        tuple: (features, full_ids_cpu, prompt_len)
            - features: Concatenated feature vector from specified layers
            - full_ids_cpu: Full token IDs (prompt + 1 generated token) on CPU
            - prompt_len: Length of the prompt in tokens
    """
    enc = tok(chat_text, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    
    # Generate exactly 1 token greedily
    out = model.generate(
        **enc,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full_ids = out.sequences[0]  # Shape: [prompt_len + 1]
    
    # Run forward pass to get hidden states (with the 1 generated token in context)
    out_hs = model(full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
    
    # Extract features at prompt_len - 1 (last token of prompt, with 1 generated token context)
    features = []
    for layer in layers:
        hs = out_hs.hidden_states[layer]
        token_idx = prompt_len - 1
        
        # Handle edge cases
        if token_idx >= hs.shape[1]:
            token_idx = hs.shape[1] - 1
        if token_idx < 0:
            token_idx = 0
        
        features.append(hs[0, token_idx, :].float().detach().cpu().numpy())
    
    return np.concatenate(features), full_ids.detach().cpu(), prompt_len

# ============================================================
# Dataset Building
# ============================================================

def build_dataset_once(tok, model, tasks, cfg, family: str):
    """
    Sample ONCE for all methods, then extract features using different methods.
    
    Uses batch generation (num_return_sequences) and parallel code execution
    for significant speedup over sequential processing.
    
    For TBG: Uses the new 1-token extraction method (generate 1 token, then extract).
    For SLT: Extracts features after full greedy generation.
    
    Args:
        tok: Tokenizer
        model: Language model
        tasks: List of task dictionaries (already mapped via get_task_fields)
        cfg: Experiment configuration
        family: Model family (e.g., "llama", "deepseek", "qwen-coder-instruct")
    
    Returns:
        List of row dictionaries with task_id, semantic_entropy, pass_at_1, and features
    """
    rows = []
    
    for ex in tqdm(tasks, desc="Building dataset", unit="task"):
        task_id = ex["task_id"]
        prompt_src = ex["prompt"]
        test_src = ex["test"]
        entry_point = ex["entry_point"]
        
        user_prompt = prompt_src
        chat_text = build_chat_text(tok, user_prompt, family=family)
        
        # 1) Batch sample M completions at once (GPU parallel via num_return_sequences)
        batch_results = batch_sample_completions(
            tok, model, chat_text,
            num_samples=cfg["M_samples"],
            max_new_tokens=cfg["sample_max_new_tokens"],
            temperature=cfg["sample_temperature"],
            top_p=cfg["sample_top_p"],
        )
        
        # Extract code from each sample
        codes = [extract_code(r["text"]) for r in batch_results]
        
        # 2) Run semantic signature checks in parallel (CPU parallel via ThreadPoolExecutor)
        sig_results = parallel_semantic_signatures(
            prompt_src, test_src, entry_point, codes,
            timeout_s=cfg["test_timeout_s"],
            max_workers=cfg.get("parallel_workers", 5),
            cluster_method=cfg.get("cluster_method", "trace_hash"),
        )
        
        # Combine results
        samples = []
        for r, (sig, ok) in zip(batch_results, sig_results):
            samples.append({"sig": sig, "ok": ok, "len_norm_logp": r["len_norm_logp"]})
        
        # Compute semantic entropy
        p_sample = softmax_probs_from_len_norm_logps([s["len_norm_logp"] for s in samples])
        cluster_mass = {}
        for s, p in zip(samples, p_sample):
            cluster_mass[s["sig"]] = cluster_mass.get(s["sig"], 0.0) + float(p)
        
        cluster_probs = np.array(list(cluster_mass.values()), dtype=np.float64)
        if cluster_probs.sum() <= 0 or not np.isfinite(cluster_probs.sum()):
            cluster_probs = np.ones_like(cluster_probs) / max(len(cluster_probs), 1)
        else:
            cluster_probs = cluster_probs / cluster_probs.sum()
        
        sem_ent = entropy_nats(cluster_probs)
        
        # 2) Greedy completion (for SLT features and pass@1)
        full_ids, prompt_len = greedy_generate_full_ids_and_prompt_len(
            tok, model, chat_text, max_new_tokens=cfg["greedy_max_new_tokens"]
        )
        gen_ids = full_ids[prompt_len:]
        greedy_text = tok.decode(gen_ids, skip_special_tokens=True)
        greedy_code = extract_code(greedy_text)
        _, pass_at_1 = semantic_signature(prompt_src, test_src, entry_point, greedy_code, timeout_s=cfg["test_timeout_s"])
        
        # 3) Extract features using SLT and TBG methods
        features_dict = {}
        
        # TBG: Use new 1-token extraction method
        # Generate 1 token, then extract features at prompt_len - 1
        if "TBG" in cfg["feature_methods"]:
            tbg_feat, _, _ = extract_tbg_features_one_token(
                tok, model, chat_text, cfg["layers"]
            )
            features_dict["TBG"] = tbg_feat
        
        # SLT: Extract from full generation (second-to-last token)
        if "SLT" in cfg["feature_methods"]:
            slt_feat = extract_features_multi_method(
                tok, model, full_ids, prompt_len, cfg["layers"], "SLT"
            )
            features_dict["SLT"] = slt_feat
        
        rows.append({
            "task_id": task_id,
            "semantic_entropy": float(sem_ent),
            "pass_at_1": int(pass_at_1),
            "prompt_len": prompt_len,
            **features_dict
        })
    
    return rows

def make_labels(semantic_entropy_values, label_mode="median"):
    """Create binary labels from semantic entropy."""
    sE = np.array(semantic_entropy_values, dtype=np.float64)
    
    if label_mode == "q25q75":
        q25, q75 = np.quantile(sE, [0.25, 0.75])
        keep = (sE <= q25) | (sE >= q75)
        y = (sE[keep] >= float(q75)).astype(int)
        return y, keep, float(q75)
    else:  # median
        thr = float(np.nanmedian(sE))
        y = (sE > thr).astype(int)
        keep = np.ones_like(y, dtype=bool)
        return y, keep, thr


def classifier_score(model, X_scaled):
    """
    Unified uncertainty score:
    - classifiers: P(high entropy)
    - regressors: predicted semantic entropy
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    return model.predict(X_scaled)

# ============================================================
# Save Dataset Splits
# ============================================================

def save_dataset_splits(df_use, y, indices_train, indices_val, indices_test, output_dir, cfg):
    """
    Save train/val/test task IDs to CSV files with dataset metadata.
    
    Args:
        df_use: DataFrame with task data
        y: Labels
        indices_train: Train split indices
        indices_val: Validation split indices
        indices_test: Test split indices
        output_dir: Output directory path
        cfg: Experiment configuration (for dataset metadata)
    
    Returns:
        tuple: (train_task_ids, val_task_ids, test_task_ids)
    """
    # Create dataset-specific split directory
    dataset_safe_name = get_dataset_safe_name(cfg.get("dataset_name", "unknown"))
    split_dir = output_dir / "DatasetSplit" / dataset_safe_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract task IDs for each split
    train_task_ids = df_use.iloc[indices_train]["task_id"].tolist()
    val_task_ids = df_use.iloc[indices_val]["task_id"].tolist()
    test_task_ids = df_use.iloc[indices_test]["task_id"].tolist()
    
    # Save to CSV
    train_df = pd.DataFrame({"task_id": train_task_ids})
    val_df = pd.DataFrame({"task_id": val_task_ids})
    test_df = pd.DataFrame({"task_id": test_task_ids})
    
    train_df.to_csv(split_dir / "train_tasks.csv", index=False)
    val_df.to_csv(split_dir / "val_tasks.csv", index=False)
    test_df.to_csv(split_dir / "test_tasks.csv", index=False)
    
    # Save summary with dataset metadata
    summary = {
        "dataset_name": cfg.get("dataset_name", "unknown"),
        "dataset_split": cfg.get("split", "unknown"),
        "prompt_field": cfg.get("prompt_field", "prompt"),
        "seed": cfg.get("seed", 42),
        "total_tasks": len(df_use),
        "train_count": len(train_task_ids),
        "val_count": len(val_task_ids),
        "test_count": len(test_task_ids),
        "train_ratio": len(train_task_ids) / len(df_use) if len(df_use) > 0 else 0,
        "val_ratio": len(val_task_ids) / len(df_use) if len(df_use) > 0 else 0,
        "test_ratio": len(test_task_ids) / len(df_use) if len(df_use) > 0 else 0,
    }
    
    with open(split_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Dataset splits saved to {split_dir}/")
    print(f"   - Dataset: {cfg.get('dataset_name', 'unknown')} ({cfg.get('split', 'unknown')})")
    print(f"   - train_tasks.csv: {len(train_task_ids)} tasks")
    print(f"   - val_tasks.csv: {len(val_task_ids)} tasks")
    print(f"   - test_tasks.csv: {len(test_task_ids)} tasks")
    print(f"   - split_summary.json: Split statistics")
    
    return train_task_ids, val_task_ids, test_task_ids

# ============================================================
# Main Experiment
# ============================================================

def get_dataset_safe_name(dataset_name: str) -> str:
    """Convert dataset name to a safe directory/file name."""
    # Replace common separators and special chars
    safe_name = dataset_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    # Common shortcuts for known datasets
    if "bigcodebench" in safe_name.lower():
        return "bigcodebench"
    elif "humaneval" in safe_name.lower():
        return "humaneval"
    return safe_name


def run_experiment():
    """Train SEP probes (SLT/TBG features with MLP/LogReg classifiers) for all models."""
    cfg = get_experiment_config()
    set_seed(cfg["seed"])
    if cfg.get("cluster_method") == "symbolic_execution":
        validate_symbolic_setup()
    
    start_time = time.time()
    
    # Create dataset-specific output directories
    output_dir = Path(__file__).parent
    dataset_safe_name = get_dataset_safe_name(cfg["dataset_name"])
    
    # Probes are saved under saved_probes/{dataset_name}/
    probes_dir = output_dir / "saved_probes" / dataset_safe_name
    probes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset: {cfg['dataset_name']}")
    print(f"Output subdirectory: {dataset_safe_name}/")
    
    # Load HuggingFace token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with model access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Load dataset (same for all models)
    print(f"\nLoading dataset: {cfg['dataset_name']} ({cfg['split']})...")
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    
    # Map dataset fields to standard format
    prompt_field = cfg.get("prompt_field", "prompt")
    raw_tasks = [ds[i] for i in range(len(ds))]
    tasks = [get_task_fields(t, cfg["dataset_name"], prompt_field) for t in raw_tasks]
    
    if cfg["limit_tasks"]:
        tasks = tasks[:cfg["limit_tasks"]]
    
    n_tasks = len(tasks)
    print(f"\n{'='*80}")
    print("PROBE TRAINING: SLT and TBG features with MLP and LogReg classifiers")
    print(f"{'='*80}")
    print(f"Dataset: {cfg['dataset_name']} ({cfg['split']})")
    print(f"Prompt field: {prompt_field}")
    print(f"Models: {len(cfg['models'])}")
    for family, size, model_id in cfg["models"]:
        print(f"  - {model_id} ({family})")
    print(f"Tasks: {n_tasks}")
    print(f"M_samples: {cfg['M_samples']}")
    print(f"Feature methods: {cfg['feature_methods']}")
    print(f"TBG extraction: 1-token generation (new method)")
    print(f"Classifiers: {cfg['classifiers']}")
    print(f"Skip existing datasets: {cfg.get('skip_existing_dataset', True)}")
    print(f"{'='*80}\n")
    
    # Results storage (across all models)
    all_results = []
    split_saved = False
    
    # Train probes for each model
    for model_idx, (family, size_bucket, model_id) in enumerate(cfg["models"]):
        print(f"\n{'='*80}")
        print(f"MODEL {model_idx + 1}/{len(cfg['models'])}: {model_id}")
        print(f"Family: {family}")
        print(f"{'='*80}\n")
        
        # Check if dataset already exists (can skip building if configured)
        model_name_safe = model_id.replace("/", "_")
        dataset_file = output_dir / f"probe_dataset_{dataset_safe_name}_{model_name_safe}.csv"
        
        if dataset_file.exists() and cfg.get("skip_existing_dataset", True):
            # Load existing dataset instead of building
            print(f"✅ Dataset already exists: {dataset_file.name}")
            print(f"   Loading from file (skipping feature extraction)...")
            print(f"   Set 'skip_existing_dataset': False to force rebuild")
            df = pd.read_csv(dataset_file)
            
            # Parse feature arrays from CSV strings
            arrays_truncated = False
            for feature_method in cfg["feature_methods"]:
                if feature_method in df.columns:
                    parsed_arrays = df[feature_method].apply(parse_array_string)
                    # Check if any arrays are truncated (None values)
                    if parsed_arrays.isna().any() or parsed_arrays.apply(lambda x: x is None).any():
                        arrays_truncated = True
                        print(f"   ⚠️  Feature arrays are truncated in CSV (cannot recover)")
                        break
                    df[feature_method] = parsed_arrays
            
            if arrays_truncated:
                # Force rebuild if arrays are truncated
                print(f"   Forcing rebuild of dataset...")
                df = None  # Signal to rebuild
            else:
                print(f"✅ Loaded {len(df)} examples from existing dataset")
                print(f"   Semantic entropy range: {df['semantic_entropy'].min():.4f} to {df['semantic_entropy'].max():.4f}")
        else:
            df = None  # Signal to build new dataset
        
        # Build dataset if needed (not loaded or truncated)
        if df is None:
            print(f"Building new dataset...")
            
            # Load model
            print(f"⏱️  Loading model {model_id}...")
            model_start = time.time()
            tok, model = load_model(model_id, HF_TOKEN)
            if tok is None:
                print(f"❌ Failed to load {model_id}, skipping...")
                continue
            model_time = time.time() - model_start
            print(f"✅ Model loaded in {model_time:.1f}s")
            
            print(f"\nStep 1: Building dataset for {model_id}...")
            print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            dataset_start = time.time()
            rows = build_dataset_once(tok, model, tasks, cfg, family=family)
            dataset_time = time.time() - dataset_start
            df = pd.DataFrame(rows)
            
            print(f"\n✅ Collected {len(df)} examples in {dataset_time/60:.1f} minutes ({dataset_time:.0f}s)")
            print(f"   Semantic entropy range: {df['semantic_entropy'].min():.4f} to {df['semantic_entropy'].max():.4f}")
            
            # Save dataset for this model (includes dataset name for differentiation)
            # Convert numpy arrays to lists to prevent truncation in CSV
            df_save = df.copy()
            for feature_method in cfg["feature_methods"]:
                if feature_method in df_save.columns:
                    df_save[feature_method] = df_save[feature_method].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )
            df_save.to_csv(dataset_file, index=False)
            print(f"✅ Dataset saved to {dataset_file}")
            
            # Cleanup model
            del model, tok
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create binary labels once (used by classification probes and threshold tuning)
        y_full, keep_mask, thr = make_labels(df["semantic_entropy"].values, cfg["label_mode"])
        df_use = df[keep_mask].reset_index(drop=True)
        y_class = y_full
        y_reg = df_use["semantic_entropy"].values.astype(np.float64)
        
        print(f"\nLabel distribution (classification): y0={np.sum(y_class==0)}, y1={np.sum(y_class==1)}")
        print(f"Semantic entropy threshold: {thr:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Step 2: Training probes for {model_id}...")
        print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        training_start = time.time()
        
        # Train probes for this model
        for feature_method in cfg["feature_methods"]:
            print(f"\n{'─'*80}")
            print(f"Feature Method: {feature_method}")
            print(f"{'─'*80}")
            
            # Extract features for this method
            X = np.stack(df_use[feature_method].values).astype(np.float32)
            print(f"Feature dimensions: {X.shape[1]}")
            
            # Split data - use stratified if possible, else fall back to regular split
            indices = np.arange(len(X))
            min_class_count = min(np.sum(y_class == 0), np.sum(y_class == 1))
            
            # Need at least 2 samples per class for stratified split with 70/15/15
            use_stratify = min_class_count >= 4
            if not use_stratify:
                print(f"  ⚠️  Small dataset ({min_class_count} samples in minority class), using non-stratified split")
            
            try:
                idx_train, idx_tmp, y_train_class, y_tmp_class = train_test_split(
                    indices, y_class, test_size=0.30, random_state=cfg["seed"], 
                    stratify=y_class if use_stratify else None
                )
                idx_val, idx_test, y_val_class, y_test_class = train_test_split(
                    idx_tmp, y_tmp_class, test_size=0.50, random_state=cfg["seed"], 
                    stratify=y_tmp_class if use_stratify and min(np.sum(y_tmp_class == 0), np.sum(y_tmp_class == 1)) >= 2 else None
                )
            except ValueError as e:
                # Fall back to non-stratified split if stratified fails
                print(f"  ⚠️  Stratified split failed ({e}), using non-stratified split")
                idx_train, idx_tmp, y_train_class, y_tmp_class = train_test_split(
                    indices, y_class, test_size=0.30, random_state=cfg["seed"]
                )
                idx_val, idx_test, y_val_class, y_test_class = train_test_split(
                    idx_tmp, y_tmp_class, test_size=0.50, random_state=cfg["seed"]
                )
            y_train_reg = y_reg[idx_train]
            y_val_reg = y_reg[idx_val]
            y_test_reg = y_reg[idx_test]
            
            # Save dataset splits (only once, using first model's first method's split)
            if not split_saved:
                save_dataset_splits(df_use, y_class, idx_train, idx_val, idx_test, output_dir, cfg)
                split_saved = True
            
            X_train = X[idx_train]
            X_val = X[idx_val]
            X_test = X[idx_test]
            
            print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            # Train entropy regression probes
            for classifier_type in cfg["classifiers"]:
                print(f"\n  Training {classifier_type.upper()} probe...")
                
                if classifier_type == "linreg":
                    clf = LinearRegression()
                    clf.fit(X_train_s, y_train_reg)
                    y_pred = clf.predict(X_test_s)
                    rmse = float(np.sqrt(mean_squared_error(y_test_reg, y_pred)))
                    r2 = float(r2_score(y_test_reg, y_pred))
                    metric_summary = {"test_rmse": rmse, "test_r2": r2}
                else:
                    raise ValueError(
                        f"Unsupported probe type '{classifier_type}'. "
                        "Only 'linreg' is supported in this strict entropy-regression setup."
                    )
                print(f"    Test RMSE:     {metric_summary['test_rmse']:.4f}")
                print(f"    Test R2:       {metric_summary['test_r2']:.4f}")
                
                # Compute recommended threshold.
                # Classification probes: median P(high entropy)
                # Regression probes: median predicted semantic entropy
                all_scores = np.concatenate([
                    classifier_score(clf, X_val_s),
                    classifier_score(clf, X_test_s),
                ])
                recommended_threshold = float(np.percentile(all_scores, 50))
                
                # Save probe (with model identifier and classifier type)
                model_name_safe = model_id.replace("/", "_")
                probe_name = f"{model_name_safe}_{feature_method}_{classifier_type}"
                probe_dir = probes_dir / probe_name
                probe_dir.mkdir(exist_ok=True)
                
                # Save probe data
                probe_data = {
                    "scaler": scaler,
                    "classifier": clf,
                    "threshold": thr,
                    "recommended_threshold": recommended_threshold,
                    "probe_kind": "regression" if classifier_type == "linreg" else "classification",
                }
                
                probe_pkl_path = probe_dir / "probe.pkl"
                with open(probe_pkl_path, "wb") as f:
                    pickle.dump(probe_data, f)
                
                # Save metadata
                probe_metadata = {
                    "model_id": model_id,
                    "model_family": family,
                    "feature_method": feature_method,
                    "classifier": classifier_type,
                    **metric_summary,
                    "feature_dim": int(X.shape[1]),
                    "layers": cfg["layers"],
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "semantic_entropy_threshold": float(thr),
                    "recommended_threshold": float(recommended_threshold),
                    "M_samples": cfg["M_samples"],
                    "label_mode": cfg["label_mode"],
                    # Dataset metadata
                    "dataset_name": cfg["dataset_name"],
                    "dataset_split": cfg["split"],
                    "prompt_field": cfg.get("prompt_field", "prompt"),
                    # TBG extraction method
                    "tbg_extraction": "one_token" if feature_method == "TBG" else "n/a",
                }
                
                probe_json_path = probe_dir / "probe_metadata.json"
                with open(probe_json_path, "w") as f:
                    json.dump(probe_metadata, f, indent=2)
                
                print(f"    ✅ Probe saved to: {probe_dir}/")
                
                all_results.append({
                    "model_id": model_id,
                    "model_family": family,
                    "feature_method": feature_method,
                    "classifier": classifier_type,
                    **metric_summary,
                    "feature_dim": X.shape[1],
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                    "n_test": len(X_test),
                    "recommended_threshold": recommended_threshold,
                    "probe_path": str(probe_dir),
                })
        
        training_time = time.time() - training_start
        print(f"\n✅ Training completed for {model_id} in {training_time/60:.1f} minutes ({training_time:.0f}s)")
    
    # Final summary across all models
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY (All Models)")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(all_results)
    if "test_accuracy" in results_df.columns:
        sort_cols = ["model_id", "test_accuracy"]
        results_df["test_accuracy"] = results_df["test_accuracy"].fillna(-1.0)
        results_df = results_df.sort_values(sort_cols, ascending=[True, False])
    else:
        results_df = results_df.sort_values(["model_id"])
    
    print(results_df.to_string(index=False))
    
    # Save results (includes dataset name for differentiation)
    results_file = output_dir / f"probe_results_{dataset_safe_name}.json"
    results_df.to_json(results_file, indent=2, orient="records")
    print(f"\n✅ Results saved to {results_file}")
    
    csv_file = output_dir / f"probe_results_{dataset_safe_name}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"✅ Results saved to {csv_file}")
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"TOTAL TIME:           {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"Completed at:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return results_df

if __name__ == "__main__":
    results = run_experiment()

