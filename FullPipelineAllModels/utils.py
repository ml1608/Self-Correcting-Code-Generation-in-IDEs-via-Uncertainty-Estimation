import torch


SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function so that it passes the tests. "
    "Return only Python code, no explanation."
)


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

    # sep_probe can be (scaler, clf, threshold) or (scaler, clf, threshold, feature_method)
    if len(sep_probe) == 4:
        scaler, clf, threshold, feature_method = sep_probe
    else:
        scaler, clf, threshold = sep_probe
        feature_method = "SLT"  # Default to SLT for backward compatibility

    # Reconstruct prompt context
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
        # SLT: Extract from prompt + generated code (matches training)
        full_text = chat_text + generated_code
        input_ids = tok(full_text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = tok(chat_text, return_tensors="pt").input_ids.shape[1]
        full_ids_cpu = input_ids.detach().cpu()

        if feature_method == "SLT" and _IMPORTED_FUNCTIONS:
            # Use SLT extraction (backward compatibility)
            feat = extract_slt_vec_multi_layer(tok, model, full_ids_cpu, layers=layers)
        else:
            # Use multi-method extraction (supports both SLT and TBG)
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
) -> str:
    """
    Generate a single code sample.

    Returns:
        Generated code string
    """
    user_prompt = prompt + "\n\n# Your code below:\n"
    chat_text = build_chat_text(tok, user_prompt)

    enc = tok(chat_text, return_tensors="pt").to(model.device)

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

    out = model.generate(**gen_kwargs)
    gen_ids = out[0][enc["input_ids"].shape[1] :]
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)
    code = extract_code(gen_text)

    return code
