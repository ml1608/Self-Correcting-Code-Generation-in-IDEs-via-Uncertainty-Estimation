import io
import os
import re
import json
import signal
import hashlib
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# ============================================================
# Dataset Field Mapping
# ============================================================

def get_task_fields(task: dict, dataset_name: str, prompt_field: str = "instruct_prompt") -> dict:
    """
    Map dataset-specific task fields to a standardized format.
    
    This helper allows consistent access to task data regardless of whether
    the source is BigCodeBench or HumanEval.
    
    Args:
        task: The task dictionary from the dataset
        dataset_name: Name of the dataset (e.g., "bigcode/bigcodebench", "openai_humaneval")
        prompt_field: Which prompt field to use (for BigCodeBench: "instruct_prompt" or "complete_prompt")
    
    Returns:
        dict with standardized keys: task_id, prompt, test, entry_point, code_prompt, canonical_solution
    """
    if "bigcodebench" in dataset_name.lower():
        # BigCodeBench provides both instruction-only and code prompts.
        # Use complete_prompt when available to ensure the function signature is present.
        prompt = task.get(prompt_field, "")
        if prompt_field == "instruct_prompt" and task.get("complete_prompt"):
            prompt = task["complete_prompt"]
        return {
            "task_id": task["task_id"],
            "prompt": prompt,
            "test": task["test"],
            "entry_point": task["entry_point"],
            # code_prompt = imports + bare function signature (no docstring).
            # Used by evaluate_completion's calibrated approach: code_prompt + "\n    pass\n"
            # is prepended so the function always has a valid body.
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


SYSTEM_PROMPT = (
    # IMPORTANT: This MUST match the system prompt used in train_probes.py
    "You are a strict coding assistant. Output only valid Python code for the function, no explanations."
)
SCRIPT_DIR = Path(__file__).parent
PROBES_DIR = SCRIPT_DIR / "saved_probes"
PROBES_DIR_BIGCODEBENCH = SCRIPT_DIR.parent / "Dataset+Probes" / "saved_probes" / "bigcodebench"
ADA_DEC_THRESHOLDS_PATH = SCRIPT_DIR / "ada_dec_thresholds.json"

# Mapping from full model IDs to JSON keys
MODEL_ID_TO_THRESHOLD_KEY = {
    "deepseek-ai/deepseek-coder-1.3b-instruct": "deepseek-1.3b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "qwen2.5-coder-3b",
}

def get_token_entropy_threshold(model_id: str, default: float = 0.3) -> float:
    """
    Get model-specific token entropy threshold for adaptive decoding.
    
    Args:
        model_id: Full model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        default: Default threshold if model not found
        
    Returns:
        Token entropy threshold for the model
    """
    if not ADA_DEC_THRESHOLDS_PATH.exists():
        print(f"⚠️  Token entropy thresholds file not found: {ADA_DEC_THRESHOLDS_PATH}")
        return default
    
    try:
        with open(ADA_DEC_THRESHOLDS_PATH, "r") as f:
            thresholds = json.load(f)
        
        # Get the key for this model
        threshold_key = MODEL_ID_TO_THRESHOLD_KEY.get(model_id)
        if threshold_key is None:
            print(f"⚠️  No threshold mapping for model: {model_id}, using default: {default}")
            return default
        
        if threshold_key not in thresholds:
            print(f"⚠️  Threshold key '{threshold_key}' not found in JSON, using default: {default}")
            return default
        
        threshold = thresholds[threshold_key]
        print(f"✅ Loaded token entropy threshold for {model_id}: {threshold}")
        return threshold
        
    except Exception as e:
        print(f"⚠️  Error loading token entropy thresholds: {e}, using default: {default}")
        return default

def get_dataset_safe_name(dataset_name: str) -> str:
    """Convert a HuggingFace dataset name to a safe directory name."""
    if "bigcodebench" in dataset_name.lower():
        return "bigcodebench"
    if "humaneval" in dataset_name.lower():
        return "humaneval"
    return dataset_name.replace("/", "_").replace(":", "_")


def get_artifact_paths(dataset_name: str) -> dict:
    """
    Return the standard probe/threshold/split paths for a given dataset.

    These paths mirror the layout produced by train_probes.py and thresh_tune.py
    under Dataset+Probes/.
    """
    safe_name = get_dataset_safe_name(dataset_name)
    dataset_probes_dir = SCRIPT_DIR.parent / "Dataset+Probes"
    return {
        "probes_dir": dataset_probes_dir / "saved_probes" / safe_name,
        "threshold_csv": dataset_probes_dir / "threshold_analysis_plots" / safe_name / "threshold_recommendations.csv",
        "split_dir": dataset_probes_dir / "DatasetSplit" / safe_name,
    }


def infer_model_family(model_id: str) -> str:
    """
    Infer the model family string from a HuggingFace model ID.

    The family string controls chat-template formatting. For unknown models
    the tokenizer's built-in chat template is used as a fallback.
    """
    m = model_id.lower()
    if "llama" in m:
        return "llama"
    if "qwen" in m:
        return "qwen-coder-instruct"
    if "deepseek" in m:
        return "deepseek"
    if "mistral" in m:
        return "mistral"
    if "gemma" in m:
        return "gemma"
    if "phi" in m:
        return "phi"
    return "unknown"


def get_probe_path(model_id: str, feature_method: str, classifier: str,
                   probes_dir: Path = None) -> Path:
    """Get probe path for a given model, feature method, and classifier.

    Args:
        model_id: Full HuggingFace model ID.
        feature_method: "SLT" or "TBG".
        classifier: "mlp" or "logreg".
        probes_dir: Directory containing probe subdirectories.
                    Defaults to the BigCodeBench probes directory for
                    backwards compatibility.
    """
    if probes_dir is None:
        probes_dir = PROBES_DIR_BIGCODEBENCH
    model_name_safe = model_id.replace("/", "_")
    probe_dir_name = f"{model_name_safe}_{feature_method}_{classifier}"
    return probes_dir / probe_dir_name

def build_chat_text(tok, user_prompt: str):
    """Build chat-formatted text for Llama."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tok, "chat_template", None) not in (None, ""):
        return tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return f"[SYSTEM] {SYSTEM_PROMPT}\n[USER] {user_prompt}\n[ASSISTANT]\n"

def extract_code(text: str) -> str:
    """Extract Python code from model output."""
    blocks = re.findall(
        r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    code = blocks[-1].strip() if blocks else text.strip()
    code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```\s*$", "", code)
    return code


@torch.inference_mode()
def extract_features_multi_method(
    tok,
    model,
    full_ids_cpu: torch.Tensor,
    prompt_len: int,
    layers: list = [-3, -2, -1],
    method: str = "SLT",
):
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
    out = model(full_ids, output_hidden_states=True, use_cache=True)

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
    full_ids_cpu: Optional[torch.Tensor] = None,
    prompt_len: Optional[int] = None,
) -> float:
    """
    Estimate uncertainty using SEP probe.

    IMPORTANT: For accurate predictions, pass full_ids_cpu and prompt_len from the actual
    generation (via generate_one_sample with return_ids=True). If not provided, the function
    will reconstruct the sequence by tokenizing, which may produce different hidden states.

    Args:
        tok: Tokenizer
        model: Language model
        prompt: The code generation prompt (raw, without "Your code below")
        generated_code: The generated code (used if full_ids_cpu not provided)
        sep_probe: (scaler, clf, threshold, feature_method) tuple
        layers: Layers to extract features from
        full_ids_cpu: Optional - actual token IDs from generation (CPU tensor)
        prompt_len: Optional - length of prompt in tokens (required if full_ids_cpu provided)

    Returns:
        Probability of high semantic entropy (0-1), higher = more uncertain
    """
    if sep_probe is None:
        return 0.5  # Default uncertainty if probe not available

    # sep_probe can be (scaler, clf, threshold) or (scaler, clf, threshold, feature_method)
    if len(sep_probe) == 4:
        scaler, clf, threshold, feature_method = sep_probe
    else:
        scaler, clf, threshold = sep_probe
        feature_method = "SLT"  # Default to SLT for backward compatibility

    # If actual token IDs are provided, use them (PREFERRED - matches training)
    if full_ids_cpu is not None and prompt_len is not None:
        # Use actual generation output - matches training exactly
        feat = extract_features_multi_method(
            tok, model, full_ids_cpu, prompt_len, layers=layers, method=feature_method
        )
    else:
        # Fallback: reconstruct sequence (may not match training exactly)
        # This path should be avoided when possible
        user_prompt = prompt + "\n\n# Your code below:\n"
        chat_text = build_chat_text(tok, user_prompt)

        # Extract features using the specified method
        # CRITICAL: TBG must be extracted from prompt ONLY (before generation)
        # SLT must be extracted from prompt + generated code (after generation)
        if feature_method == "TBG":
            # TBG: Extract from prompt only (matches training)
            input_ids = tok(chat_text, return_tensors="pt").input_ids.to(model.device)
            prompt_len = input_ids.shape[1]
            full_ids_cpu = input_ids.detach().cpu()
            feat = extract_features_multi_method(
                tok, model, full_ids_cpu, prompt_len, layers=layers, method=feature_method
            )
        else:
            # SLT: Extract from prompt + generated code
            # WARNING: This may not match training exactly due to tokenization differences
            full_text = chat_text + generated_code
            input_ids = tok(full_text, return_tensors="pt").input_ids.to(model.device)
            prompt_len = tok(chat_text, return_tensors="pt").input_ids.shape[1]
            full_ids_cpu = input_ids.detach().cpu()

            feat = extract_features_multi_method(
                tok,
                model,
                full_ids_cpu,
                prompt_len,
                layers=layers,
                method=feature_method,
            )

    # Predict uncertainty using SEP probe
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    prob_high_entropy = clf.predict_proba(feat_scaled)[
        0, 1
    ]  # Probability of high semantic entropy

    return float(prob_high_entropy)


@torch.inference_mode()
def generate_one_sample(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    return_ids: bool = False,
) -> str:
    """
    Generate a single code sample.

    Args:
        tok: Tokenizer
        model: Language model
        prompt: The code generation prompt (raw, without "Your code below")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling parameter
        return_ids: If True, return (code, full_ids, prompt_len) tuple

    Returns:
        If return_ids=False: Generated code string
        If return_ids=True: (code, full_ids_cpu, prompt_len) tuple
    """
    user_prompt = prompt + "\n\n# Your code below:\n"
    chat_text = build_chat_text(tok, user_prompt)

    enc = tok(chat_text, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    # Build generation kwargs
    gen_kwargs = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "pad_token_id": tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
    }

    # For greedy (temp=0.0), use do_sample=False
    # For sampling (temp>0), use do_sample=True with temperature
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False
        # Explicitly set neutral values to override model's generation_config defaults
        # (prevents warnings about temperature/top_p being set with do_sample=False)
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    out = model.generate(**gen_kwargs)
    full_ids = out[0]
    gen_ids = full_ids[prompt_len:]
    # Use skip_special_tokens=True to remove model-specific tokens like <|EOT|> (DeepSeek)
    # These tokens cause invalid syntax when included in the extracted code
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    code = extract_code(gen_text)

    if return_ids:
        return code, full_ids.detach().cpu(), prompt_len
    return code
