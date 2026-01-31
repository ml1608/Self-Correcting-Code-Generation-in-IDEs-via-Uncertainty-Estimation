#!/usr/bin/env python3
"""
Probe Training Experiment: Compare SLT, TBG, and Neural Network Methods

This script:
1. Samples code ONCE (M_samples=20) for all methods
2. Extracts features using different methods (SLT, TBG, LAST)
3. Trains different classifiers (Random Forest, MLP, Deep NN)
4. Compares all combinations to find best accuracy
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
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError
from getpass import getpass

# ============================================================
# Configuration
# ============================================================

def get_experiment_config():
    return {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "dataset_name": "openai_humaneval",
        "split": "test",
        "limit_tasks": 30,  # Start small for testing, set to None for full 164
        
        # Sampling (done ONCE for all methods)
        "M_samples": 20,  # Increased for better semantic entropy
        
        "sample_max_new_tokens": 256,
        "sample_temperature": 0.7,
        "sample_top_p": 0.95,
        "greedy_max_new_tokens": 256,
        "test_timeout_s": 10,
        "seed": 42,
        
        # Feature extraction methods to test
        "feature_methods": ["SLT", "TBG", "LAST"],  # SLT=-2, TBG=prompt_len-1, LAST=-1
        
        # Layers to use
        "layers": [-3, -2, -1],
        
        # Classifiers to test
        "classifiers": ["random_forest", "mlp", "deep_nn"],
        
        # Labeling
        "label_mode": "median",  # Use all examples (no filtering) - median split for binary classification
    }

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

# ============================================================
# Helper Functions
# ============================================================

def build_chat_text(tok, user_prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tok, "chat_template", None) not in (None, ""):
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"[SYSTEM] {SYSTEM_PROMPT}\n[USER] {user_prompt}\n[ASSISTANT]\n"

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
# Feature Extraction (Multiple Methods)
# ============================================================

@torch.inference_mode()
def extract_features_multi_method(tok, model, full_ids_cpu: torch.Tensor, prompt_len: int, 
                                   layers: list, method: str):
    """
    Extract features using different methods:
    - SLT: second-to-last token (index -2)
    - TBG: token before generation (prompt_len - 1)
    - LAST: last token (index -1, current baseline)
    """
    full_ids = full_ids_cpu.unsqueeze(0).to(model.device)
    out = model(full_ids, output_hidden_states=True, use_cache=False)
    
    features = []
    for layer in layers:
        hs = out.hidden_states[layer]
        
        if method == "SLT":
            # Second-to-last token
            token_idx = -2
        elif method == "TBG":
            # Token before generation (last token of prompt)
            token_idx = prompt_len - 1
        elif method == "LAST":
            # Last token (current baseline)
            token_idx = -1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle edge cases
        if token_idx < 0:
            token_idx = hs.shape[1] + token_idx  # Convert negative to positive
        
        if token_idx >= hs.shape[1]:
            token_idx = hs.shape[1] - 1  # Clamp to valid range
        
        if token_idx < 0:
            token_idx = 0  # Final safety check
        
        features.append(hs[0, token_idx, :].float().detach().cpu().numpy())
    
    return np.concatenate(features)

# ============================================================
# Deep Neural Network Classifier
# ============================================================

class DeepUncertaintyProbe(nn.Module):
    """
    Deep Neural Network for binary uncertainty classification.
    
    Architecture:
    - Input: Concatenated hidden states from multiple transformer layers
            (e.g., 3 layers × 4096 dims = 12,288 input features)
    - Hidden: Deep feedforward network with batch norm and dropout
    - Output: Single neuron with sigmoid (binary classification: high/low semantic entropy)
    
    This is BINARY classification (not multilabel):
    - Output = 0: Low semantic entropy (confident, consistent generations)
    - Output = 1: High semantic entropy (uncertain, diverse generations)
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        # Binary classification output
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_deep_nn(X_train, y_train, X_val, y_val, input_dim, epochs=200, batch_size=64, lr=0.001):
    """
    Train PyTorch deep neural network with early stopping and learning rate scheduling.
    
    Returns:
        (trained_model, best_validation_accuracy)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DeepUncertaintyProbe(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device).unsqueeze(1)
    
    # Create DataLoader for batch training
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_acc = 0
    best_model_state = None
    patience = 15  # Increased patience for deeper network
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()
            val_loss = criterion(val_outputs, y_val_t).item()
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_classifier(clf, X_test, y_test, classifier_type="random_forest"):
    """Evaluate classifier and return accuracy and AUROC."""
    if classifier_type == "deep_nn":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            probs = clf(X_test_t).cpu().numpy().flatten()
        y_pred = (probs >= 0.5).astype(int)
    else:
        probs = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
    
    return acc, auc, probs

def load_saved_probe(probe_dir: str):
    """
    Load a saved probe from disk.
    
    Returns:
        (scaler, classifier, threshold, recommended_threshold, metadata)
    """
    probe_pkl_path = os.path.join(probe_dir, "probe.pkl")
    probe_json_path = os.path.join(probe_dir, "probe_metadata.json")
    
    if not os.path.exists(probe_pkl_path):
        raise FileNotFoundError(f"Probe not found: {probe_pkl_path}")
    
    with open(probe_pkl_path, "rb") as f:
        probe_data = pickle.load(f)
    
    scaler = probe_data["scaler"]
    clf = probe_data["classifier"]
    threshold = probe_data.get("threshold", None)
    recommended_threshold = probe_data.get("recommended_threshold", 0.5)
    
    metadata = None
    if os.path.exists(probe_json_path):
        with open(probe_json_path, "r") as f:
            metadata = json.load(f)
    
    return scaler, clf, threshold, recommended_threshold, metadata

# ============================================================
# Dataset Building (Sample Once, Extract Features All Methods)
# ============================================================

def build_dataset_once(tok, model, tasks, cfg):
    """Sample ONCE for all methods, then extract features using different methods."""
    rows = []
    
    for ex in tqdm(tasks, desc="Building dataset", unit="task"):
        task_id = ex["task_id"]
        prompt_src = ex["prompt"]
        test_src = ex["test"]
        entry_point = ex["entry_point"]
        
        user_prompt = prompt_src + "\n\n# Your code below:\n"
        chat_text = build_chat_text(tok, user_prompt)
        
        # 1) Sample M completions ONCE (for semantic entropy)
        samples = []
        for _ in range(cfg["M_samples"]):
            s = sample_completion_with_len_norm_logp(
                tok, model, chat_text,
                max_new_tokens=cfg["sample_max_new_tokens"],
                temperature=cfg["sample_temperature"],
                top_p=cfg["sample_top_p"],
            )
            code = extract_code(s["text"])
            sig, ok = semantic_signature(prompt_src, test_src, entry_point, code, timeout_s=cfg["test_timeout_s"])
            samples.append({"sig": sig, "ok": ok, "len_norm_logp": s["len_norm_logp"]})
        
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
        
        # 2) Greedy completion (for features and pass@1)
        full_ids, prompt_len = greedy_generate_full_ids_and_prompt_len(
            tok, model, chat_text, max_new_tokens=cfg["greedy_max_new_tokens"]
        )
        gen_ids = full_ids[prompt_len:]
        greedy_text = tok.decode(gen_ids, skip_special_tokens=True)
        greedy_code = extract_code(greedy_text)
        _, pass_at_1 = semantic_signature(prompt_src, test_src, entry_point, greedy_code, timeout_s=cfg["test_timeout_s"])
        
        # Extract features using ALL methods (from same generation)
        features_dict = {}
        for method in cfg["feature_methods"]:
            feat = extract_features_multi_method(
                tok, model, full_ids, prompt_len, cfg["layers"], method
            )
            features_dict[method] = feat
        
        rows.append({
            "task_id": task_id,
            "semantic_entropy": float(sem_ent),
            "pass_at_1": int(pass_at_1),
            "prompt_len": prompt_len,
            **features_dict  # Add features for each method
        })
    
    return rows

def make_labels(semantic_entropy_values, label_mode="q25q75"):
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

# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    """Run the full experiment comparing all methods."""
    cfg = get_experiment_config()
    set_seed(cfg["seed"])
    
    start_time = time.time()
    
    # Load model
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\nHugging Face login (token is not printed).")
        HF_TOKEN = getpass("Paste your Hugging Face token (with Llama access): ").strip()
        if not HF_TOKEN:
            raise ValueError("Empty HF token. Please paste a valid token.")
    
    login(HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in successfully!")
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    print("\n⏱️  Loading model...")
    model_start = time.time()
    tok, model = load_model(cfg["model_id"], HF_TOKEN)
    if tok is None:
        print("Failed to load model")
        return
    model_time = time.time() - model_start
    print(f"✅ Model loaded in {model_time:.1f}s")
    
    # Load dataset
    ds = load_dataset(cfg["dataset_name"])[cfg["split"]]
    tasks = [ds[i] for i in range(len(ds))]
    if cfg["limit_tasks"]:
        tasks = tasks[:cfg["limit_tasks"]]
    
    # Estimate runtime
    n_tasks = len(tasks)
    m_samples = cfg["M_samples"]
    
    # Rough estimates (per task):
    # - 20 samples: ~3-5s each = 60-100s
    # - 1 greedy: ~1-2s
    # - Feature extraction: ~0.5s
    # - Tests: ~0.5s per sample = 10s
    # Total per task: ~75-115s
    est_per_task = 90  # seconds (conservative estimate)
    est_dataset_time = (n_tasks * est_per_task) / 60  # minutes
    
    # Training estimates (9 combinations):
    # - Random Forest: ~10-20s each = 30-60s total
    # - MLP: ~20-40s each = 60-120s total  
    # - Deep NN: ~2-5 min each (deeper network, more epochs) = 6-15 min total
    est_training_time = 15  # minutes (conservative, accounting for deeper NN)
    
    total_est = (est_dataset_time + est_training_time)  # minutes
    
    print(f"\n{'='*80}")
    print("PROBE TRAINING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Model: {cfg['model_id']}")
    print(f"Tasks: {n_tasks}")
    print(f"M_samples: {m_samples} (sampled ONCE for all methods)")
    print(f"Feature methods: {cfg['feature_methods']} ({len(cfg['feature_methods'])} methods)")
    print(f"Classifiers: {cfg['classifiers']} ({len(cfg['classifiers'])} classifiers)")
    print(f"Total combinations: {len(cfg['feature_methods']) * len(cfg['classifiers'])}")
    print(f"\n⏱️  ESTIMATED RUNTIME:")
    print(f"   Dataset building: ~{est_dataset_time:.0f} minutes")
    print(f"   Training & evaluation: ~{est_training_time:.0f} minutes")
    print(f"   TOTAL: ~{total_est:.0f} minutes ({total_est/60:.1f} hours)")
    print(f"{'='*80}\n")
    
    # Build dataset (sample once, extract features using all methods)
    print("Step 1: Building dataset (sampling once, extracting features for all methods)...")
    print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    dataset_start = time.time()
    rows = build_dataset_once(tok, model, tasks, cfg)
    dataset_time = time.time() - dataset_start
    df = pd.DataFrame(rows)
    
    print(f"\n✅ Collected {len(df)} examples in {dataset_time/60:.1f} minutes ({dataset_time:.0f}s)")
    print(f"   Semantic entropy range: {df['semantic_entropy'].min():.4f} to {df['semantic_entropy'].max():.4f}")
    print(f"   Average time per task: {dataset_time/len(tasks):.1f}s")
    
    # Cleanup model
    del model, tok
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create labels
    y_full, keep_mask, thr = make_labels(df["semantic_entropy"].values, cfg["label_mode"])
    df_use = df[keep_mask].reset_index(drop=True)
    y = y_full
    
    print(f"\nLabel distribution: y0={np.sum(y==0)}, y1={np.sum(y==1)}")
    print(f"Semantic entropy threshold: {thr:.4f}")
    
    # Results storage
    results = []
    
    # Create directory for saving probes
    probes_dir = "saved_probes"
    os.makedirs(probes_dir, exist_ok=True)
    print(f"📁 Probes will be saved to: {probes_dir}/\n")
    
    # Test all combinations
    print(f"\n{'='*80}")
    print("Step 2: Training and evaluating all combinations...")
    print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    training_start = time.time()
    
    for feature_method in cfg["feature_methods"]:
        print(f"\n{'─'*80}")
        print(f"Feature Method: {feature_method}")
        print(f"{'─'*80}")
        
        # Extract features for this method
        X = np.stack(df_use[feature_method].values).astype(np.float32)
        print(f"Feature dimensions: {X.shape[1]}")
        
        # Split data
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.30, random_state=cfg["seed"], stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=cfg["seed"], stratify=y_tmp
        )
        
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        for classifier_type in cfg["classifiers"]:
            print(f"\n  Classifier: {classifier_type}")
            
            # Train classifier
            if classifier_type == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=cfg["seed"], n_jobs=-1, verbose=0
                )
                clf.fit(X_train_s, y_train)
                
            elif classifier_type == "mlp":
                from sklearn.neural_network import MLPClassifier
                clf = MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    max_iter=1000, random_state=cfg["seed"],
                    early_stopping=True, validation_fraction=0.1, verbose=False
                )
                clf.fit(X_train_s, y_train)
                
            elif classifier_type == "deep_nn":
                clf, val_acc = train_deep_nn(
                    X_train_s, y_train, X_val_s, y_val,
                    input_dim=X_train_s.shape[1]
                )
                print(f"    Best val accuracy during training: {val_acc:.4f}")
            
            # Evaluate
            acc, auc, probs = evaluate_classifier(clf, X_test_s, y_test, classifier_type)
            
            print(f"    Test Accuracy: {acc:.4f}")
            print(f"    Test AUROC:    {auc:.4f}")
            
            # Compute recommended threshold (median of predictions on val+test)
            if classifier_type == "deep_nn":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                clf.eval()
                with torch.no_grad():
                    X_val_test = np.vstack([X_val_s, X_test_s])
                    X_val_test_t = torch.FloatTensor(X_val_test).to(device)
                    all_probs = clf(X_val_test_t).cpu().numpy().flatten()
            else:
                all_probs = np.concatenate([
                    clf.predict_proba(X_val_s)[:, 1],
                    clf.predict_proba(X_test_s)[:, 1]
                ])
            
            recommended_threshold = np.percentile(all_probs, 50)
            
            # Save probe
            probe_name = f"{feature_method}_{classifier_type}"
            probe_dir = os.path.join(probes_dir, probe_name)
            os.makedirs(probe_dir, exist_ok=True)
            
            # Save probe data
            probe_data = {
                "scaler": scaler,
                "classifier": clf,
                "threshold": thr,  # Semantic entropy threshold for labeling
                "recommended_threshold": recommended_threshold,  # For inference
            }
            
            probe_pkl_path = os.path.join(probe_dir, "probe.pkl")
            with open(probe_pkl_path, "wb") as f:
                pickle.dump(probe_data, f)
            
            # Save metadata
            probe_metadata = {
                "feature_method": feature_method,
                "classifier": classifier_type,
                "test_accuracy": float(acc),
                "test_auc": float(auc),
                "feature_dim": int(X.shape[1]),
                "layers": cfg["layers"],
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "semantic_entropy_threshold": float(thr),
                "recommended_threshold": float(recommended_threshold),
                "model_id": cfg["model_id"],
                "M_samples": cfg["M_samples"],
                "label_mode": cfg["label_mode"],
            }
            
            probe_json_path = os.path.join(probe_dir, "probe_metadata.json")
            with open(probe_json_path, "w") as f:
                json.dump(probe_metadata, f, indent=2)
            
            print(f"    ✅ Probe saved to: {probe_dir}/")
            
            results.append({
                "feature_method": feature_method,
                "classifier": classifier_type,
                "test_accuracy": acc,
                "test_auc": auc,
                "feature_dim": X.shape[1],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "recommended_threshold": recommended_threshold,
                "probe_path": probe_dir,
            })
    
    training_time = time.time() - training_start
    print(f"\n✅ Training & evaluation completed in {training_time/60:.1f} minutes ({training_time:.0f}s)")
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_accuracy", ascending=False)
    
    print(results_df.to_string(index=False))
    
    # Best combination
    best = results_df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST COMBINATION:")
    print(f"{'='*80}")
    print(f"Feature Method: {best['feature_method']}")
    print(f"Classifier:      {best['classifier']}")
    print(f"Test Accuracy:   {best['test_accuracy']:.4f}")
    print(f"Test AUROC:      {best['test_auc']:.4f}")
    print(f"Recommended Threshold: {best['recommended_threshold']:.4f}")
    print(f"Probe Path:      {best['probe_path']}")
    print(f"{'='*80}\n")
    
    # Save results
    output_file = "probe_experiment_results.json"
    results_df.to_json(output_file, indent=2, orient="records")
    print(f"✅ Results saved to {output_file}")
    
    # Also save as CSV for easy viewing
    csv_file = "probe_experiment_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"✅ Results saved to {csv_file}")
    
    print(f"\n✅ All probes saved to: {probes_dir}/")
    print(f"   Each probe includes: probe.pkl (classifier + scaler) and probe_metadata.json")
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Model loading:        {model_time:.1f}s")
    print(f"Dataset building:     {dataset_time/60:.1f} min ({dataset_time:.0f}s)")
    print(f"Training & eval:      {training_time/60:.1f} min ({training_time:.0f}s)")
    print(f"TOTAL TIME:           {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"Completed at:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return results_df

if __name__ == "__main__":
    results = run_experiment()

