#!/usr/bin/env python3
"""
Symbolic clustering helper for train_probes.py.

Contract:
  python pyexz3_cluster_helper.py --entry-point <fn_name> --timeout <sec> <module_file.py>
Output (stdout, JSON):
  {"cluster_id": "...", "passed": 0|1}

Notes:
- This helper is intentionally self-contained for Lambda execution.
- It computes:
  1) pass/fail by running `check(fn)` from the task test harness
  2) a deterministic semantic cluster id from the target function's normalized AST
     plus normalized execution outcome.
"""

import argparse
import ast
import hashlib
import io
import json
import re
import signal
from contextlib import redirect_stdout, redirect_stderr


def _normalize_text(s: str) -> str:
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"\d+\.\d+", "FLOAT", s)
    s = re.sub(r"\b\d+\b", "INT", s)
    return s


def _safe_exec_module(module_src: str):
    glb = {}
    exec(module_src, glb, glb)
    return glb


def _run_check(glb: dict, entry_point: str, timeout_seconds: int):
    f = io.StringIO()
    use_timeout = hasattr(signal, "SIGALRM")
    old_handler = None
    if use_timeout:
        def handler(signum, frame):
            raise TimeoutError(f"timeout>{timeout_seconds}s")
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_seconds)
    try:
        with redirect_stdout(f), redirect_stderr(f):
            fn = glb[entry_point]
            glb["check"](fn)
        return 1, f.getvalue(), ""
    except Exception as e:
        return 0, f.getvalue(), repr(e)
    finally:
        if use_timeout and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _get_function_source(module_src: str, entry_point: str) -> str:
    try:
        tree = ast.parse(module_src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                return ast.unparse(node)
    except Exception:
        pass
    return ""


def _ast_fingerprint(func_src: str) -> str:
    if not func_src:
        return "NOFUNC"
    try:
        tree = ast.parse(func_src)
        dumped = ast.dump(tree, annotate_fields=False, include_attributes=False)
        dumped = _normalize_text(dumped)
        return hashlib.sha256(dumped.encode("utf-8", errors="ignore")).hexdigest()[:16]
    except Exception:
        norm = _normalize_text(func_src)
        return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]


def compute_cluster(module_src: str, entry_point: str, timeout_seconds: int):
    func_src = _get_function_source(module_src, entry_point)
    ast_sig = _ast_fingerprint(func_src)

    # Robust behavior for large benchmark datasets:
    # if module import/exec fails (e.g., optional dependency like seaborn missing),
    # return a deterministic failure cluster instead of crashing probe training.
    try:
        glb = _safe_exec_module(module_src)
        passed, logs, err = _run_check(glb, entry_point, timeout_seconds)
    except Exception as e:
        passed, logs, err = 0, "", f"EXEC_ERROR:{repr(e)}"

    outcome_sig_raw = _normalize_text((logs or "") + "||" + (err or ""))
    outcome_sig = hashlib.sha256(outcome_sig_raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    cluster_id = f"AST:{ast_sig}:OUT:{outcome_sig}:PASS:{passed}"
    return {"cluster_id": cluster_id, "passed": int(passed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-point", required=False, default=None)
    parser.add_argument("--timeout", required=False, default="10")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("module_file", nargs="?")
    args = parser.parse_args()

    if args.self_check:
        print(json.dumps({"ok": True}))
        return

    if not args.entry_point:
        raise ValueError("--entry-point is required")
    if not args.module_file:
        raise ValueError("module_file is required")

    timeout_seconds = int(args.timeout)
    with open(args.module_file, "r", encoding="utf-8") as f:
        module_src = f.read()

    out = compute_cluster(module_src, args.entry_point, timeout_seconds)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
