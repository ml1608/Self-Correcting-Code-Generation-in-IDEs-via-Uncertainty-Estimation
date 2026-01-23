import time, math
import torch
import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
model.eval()


entropy_threshold = 3.5
beam_size = 3
lookahead_length = 5


@torch.no_grad()
def greedy_decode(prompt: str, max_new_tokens: int = 256) -> Tuple[str, int, float]:
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output_ids = input_ids.clone()
    start = time.time()

    for _ in range(max_new_tokens):
        logits = model(output_ids).logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1))
        output_ids = torch.cat([output_ids, torch.tensor([[next_id]], device="cuda")], dim=1)
        if next_id == tokenizer.eos_token_id:
            break

    latency = time.time() - start
    gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    return gen_text, (output_ids.shape[1] - input_ids.shape[1]), latency


@torch.no_grad()
def adaptive_decode(prompt: str, max_new_tokens: int = 256) -> Tuple[str, int, float]:
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
    output_ids = input_ids.clone()
    start_time = time.time()

    for _ in range(max_new_tokens):
        outputs = model(output_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Shannon entropy
        p = probs[0].detach().float().cpu().numpy()
        entropy = -float(np.sum(p * np.log(p + 1e-10)))

        if entropy > entropy_threshold:
            topk = torch.topk(probs, beam_size, dim=-1)
            candidate_tokens = topk.indices[0]  # (beam_size,)
            candidate_probs = topk.values[0]    # (beam_size,)

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

                    if tokenizer.decode(int(nxt)) == "\n":
                        break

                traj_len = sim_ids.shape[1] - output_ids.shape[1]
                avg_log_prob = score / max(traj_len, 1)

                if avg_log_prob > best_score:
                    best_score = avg_log_prob
                    best_token = int(token)

            next_token_id = best_token if best_token is not None else int(torch.argmax(probs))
        else:
            next_token_id = int(torch.argmax(probs))

        output_ids = torch.cat([output_ids, torch.tensor([[next_token_id]], device="cuda")], dim=1)

        if next_token_id == tokenizer.eos_token_id:
            break

    total_time = time.time() - start_time
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    return generated_text, (output_ids.shape[1] - input_ids.shape[1]), total_time



def _run_check(check_fn: Callable[[str], bool], completion: str) -> bool:
    
    try:
        return bool(check_fn(completion))
    except Exception:
        return False


def evaluate_pass_at_1(
    tasks: List[Dict[str, Any]],
    max_new_tokens: int = 256,
) -> Tuple[float, float, float]:
   
    n = len(tasks)
    if n == 0:
        return 0.0, 0.0, 0.0

    before_pass = 0
    after_pass = 0
    after_latencies = []

    for t in tasks:
        prompt = t["prompt"]
        check_fn = t["check"]

        # baseline
        base_text, _, _ = greedy_decode(prompt, max_new_tokens=max_new_tokens)
        if _run_check(check_fn, base_text):
            before_pass += 1

        # adaptive
        ada_text, _, ada_latency = adaptive_decode(prompt, max_new_tokens=max_new_tokens)
        after_latencies.append(ada_latency)
        if _run_check(check_fn, ada_text):
            after_pass += 1

    pass_before = before_pass / n
    pass_after = after_pass / n
    avg_latency_after = float(np.mean(after_latencies))

    return pass_before, pass_after, avg_latency_after



def check_is_prime(completion: str) -> bool:
   
    code = "import math\n" + completion
    g = {}
    l = {}
    exec(code, g, l)  # <-- for real eval, run in sandboxed process/container
    if "is_prime" not in l:
        return False
    f = l["is_prime"]
    tests = [
        (2, True), (3, True), (4, False), (5, True), (9, False),
        (97, True), (1, False), (0, False), (-7, False)
    ]
    for x, y in tests:
        if f(x) != y:
            return False
    return True


tasks = [
    {"prompt": "def is_prime(n):\n    # Determine if n is a prime number\n", "check": check_is_prime},
]

pass_before, pass_after, latency = evaluate_pass_at_1(tasks, max_new_tokens=100)

print(pass_before)  
print(pass_after)   
print(latency)      