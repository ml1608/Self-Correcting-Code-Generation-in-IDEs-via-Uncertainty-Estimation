import torch
import multiprocessing
import contextlib
import io
import gc
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import notebook_login


if "notebook_login" in locals():
    print("Checking Hugging Face Login...")
    notebook_login()


MODELS_TO_COMPARE = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # Official small R1 variant
]
DATASET_NAME = "openai_humaneval"
NUM_SAMPLES = 20           
T_SAMPLES = 10             # Pass@10 (10 generations per problem)
TIMEOUT_SECONDS = 3        



def clean_deepseek_output(text):
    """Removes <think> tags to extract just the code/answer."""
    # Remove content between <think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove stand-alone tags if regex missed them
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()

def get_code_samples(model_name, model, tokenizer, prompt, device, T=10):
    model.train()

    if "Qwen" in model_name or "Llama" in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Complete the python function based on the docstring. Do not explain, just write code."},
            {"role": "user", "content": prompt}
        ]
        try:
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        except:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    elif "DeepSeek" in model_name:
        
        full_prompt = f"<|User|>{prompt}\nPlease complete the Python function. Output only valid Python code.<|Assistant|>"
        inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

    generated_codes = []

    with torch.no_grad():
        for _ in range(T):
            outputs = model.generate(
                inputs,
                max_new_tokens=512, 
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            
            if "DeepSeek" in model_name:
                
                full_text = clean_deepseek_output(full_text)

            
            code = full_text
            if "```python" in full_text:
                try: code = full_text.split("```python")[1].split("```")[0]
                except: pass
            elif "```" in full_text:
                try: code = full_text.split("```")[1].split("```")[0]
                except: pass
            elif "def " in full_text:
                
                code = full_text[full_text.find("def "):]

            
            code = code.replace("Here is the code:", "").strip()
            generated_codes.append(code)

    model.eval()
    return generated_codes

def unsafe_execute(code_str, result_queue):
    
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec_globals = {}
            exec(code_str, exec_globals)
        result_queue.put("passed")
    except Exception:
        result_queue.put("failed")

def verify_code(prompt, completion, test_case, entry_point):
    
    # Construct full executable
    if completion.strip().startswith("def "):
        full_code = completion + "\n\n" + test_case + f"\ncheck({entry_point})"
    else:
        full_code = prompt + completion + "\n\n" + test_case + f"\ncheck({entry_point})"

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=unsafe_execute, args=(full_code, queue))
    p.start()
    p.join(TIMEOUT_SECONDS)

    if p.is_alive():
        p.terminate()
        p.join()
        return 0

    if not queue.empty() and queue.get() == "passed":
        return 1
    return 0

def calculate_uncertainty(samples, similarity_model, device):
    
    if not samples: return 0.0, ""
    embeddings = similarity_model.encode(samples, convert_to_tensor=True, device=device)
    # High threshold for code
    clusters = util.community_detection(embeddings, min_community_size=1, threshold=0.90)

    if not clusters: return 1.0/len(samples), samples[0]

    largest_cluster = max(clusters, key=len)
    confidence = len(largest_cluster) / len(samples)
    best_sample = samples[largest_cluster[0]]
    return confidence, best_sample


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Similarity Model...")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print("Loading HumanEval Dataset...")
    dataset = load_dataset(DATASET_NAME, split="test").select(range(NUM_SAMPLES))

    all_results = []

    for model_name in MODELS_TO_COMPARE:
        print(f"\n\n>>> EVALUATING: {model_name} <<<")

        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
            trust_remote_code=True
        )

        for item in tqdm(dataset, desc=f"Coding with {model_name.split('/')[-1]}"):
            
            samples = get_code_samples(model_name, model, tokenizer, item['prompt'], device, T=T_SAMPLES)

            
            confidence, best_sample = calculate_uncertainty(samples, sim_model, device)

            
            representative_is_correct = verify_code(item['prompt'], best_sample, item['test'], item['entry_point'])

            all_results.append({
                "Model": model_name.split("/")[-1],
                "Task": item['task_id'],
                "Confidence": confidence,
                "Correctness": representative_is_correct, # 0 or 1
                "Code": best_sample,
                "Prompt": item['prompt']
            })

        # Free Memory
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


    df = pd.DataFrame(all_results)
    print("\nGenerating Plots...")

    models = df['Model'].unique()

    for model in models:
        model_df = df[df['Model'] == model]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Histogram
        # Blue = Correct, Red = Incorrect
        sns.histplot(data=model_df[model_df['Correctness']==1], x='Confidence', color='blue', label='Correct', kde=False, bins=10, ax=axes[0], alpha=0.6)
        sns.histplot(data=model_df[model_df['Correctness']==0], x='Confidence', color='red', label='Incorrect', kde=False, bins=10, ax=axes[0], alpha=0.6)
        axes[0].set_title(f"{model}\nConfidence Distribution")
        axes[0].set_xlabel("Confidence Score (P(true))")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # Plot 2: Scatter Plot (Confidence vs Correctness)
        # Add Jitter to see points clearly
        jitter_y = model_df['Correctness'] + np.random.normal(0, 0.03, size=len(model_df))
        jitter_x = model_df['Confidence'] + np.random.normal(0, 0.01, size=len(model_df))

        sc = axes[1].scatter(jitter_x, jitter_y, c=model_df['Correctness'], cmap='coolwarm_r', alpha=0.7, edgecolors='k')
        axes[1].set_title(f"{model}\nConfidence vs Correctness")
        axes[1].set_xlabel("Confidence Score")
        axes[1].set_ylabel("Correctness (0=Fail, 1=Pass)")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Incorrect", "Correct"])

        plt.tight_layout()
        plt.show()

        # EXAMPLES ANALYSIS
        print(f"\n--- Analysis for {model} ---")

        # Case 1: Correct behavior (High Conf + Correct)
        success_case = model_df[(model_df['Correctness'] == 1) & (model_df['Confidence'] > 0.8)]
        if not success_case.empty:
            row = success_case.iloc[0]
            print(f"✅ [Good Calibration] The model was confident ({row['Confidence']:.2f}) and Correct.")
            print(f"   Task: {row['Task']}")
            print("   Reasoning: All samples were likely identical, and the logic was sound.")

        # Case 2: Failure behavior (High Conf + Incorrect)
        fail_case = model_df[(model_df['Correctness'] == 0) & (model_df['Confidence'] > 0.8)]
        if not fail_case.empty:
            row = fail_case.iloc[0]
            print(f"❌ [Overconfidence Failure] The model was confident ({row['Confidence']:.2f}) but WRONG.")
            print(f"   Task: {row['Task']}")
            print("   Reasoning: The model consistently generated the same BUGGY code. This often happens with subtle edge cases.")
        else:
            print("   (No severe overconfidence detected in this batch.)")
        print("-" * 80)

if __name__ == "__main__":
    main()