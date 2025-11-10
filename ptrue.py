import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


# Model and Dataset Configuration 
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_NAME = "boolq"
NUM_SAMPLES_FOR_DEMO = 30 # How many questions to process 

# Uncertainty Estimation Configuration
T_SAMPLES_PER_PROMPT = 10 # Number of Monte Carlo samples per question
N_BINS_FOR_ECE = 10       # Number of bins for the ECE calculation and plot
SIMILARITY_THRESHOLD = 0.8 # Threshold for semantic clustering (0.0 to 1.0)



def get_mcdo_predictions(model, tokenizer, prompt, device, T=10, max_new_tokens=10):
    """Performs Monte Carlo Dropout to get a distribution of text generations."""
    model.train() # Activate dropout layers
    generated_texts = []
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(T):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode only the newly generated part
            generated_part = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            generated_texts.append(generated_part.strip())

    model.eval() # Return to evaluation mode
    return generated_texts

def calculate_confidence_and_answer(samples, similarity_model, device):
    """Calculates semantic consistency confidence and determines the majority answer."""
    if not samples or len(samples) == 0:
        return 0.0, "N/A"

    embeddings = similarity_model.encode(samples, convert_to_tensor=True, device=device)
    # Cluster embeddings to find semantically similar answers
    clusters = util.community_detection(embeddings, min_community_size=1, threshold=SIMILARITY_THRESHOLD)

    if not clusters:
        # If no clusters form, confidence is low (each answer is unique)
        return 1.0 / len(samples), samples[0] 

    # Find the largest cluster
    largest_cluster = max(clusters, key=len)
    
    # Confidence = size of largest cluster / total samples
    confidence = len(largest_cluster) / len(samples)
    
    # The final answer is the text from the largest cluster
    majority_answer_text = samples[largest_cluster[0]]

    # Normalize the answer for yes/no task
    if 'yes' in majority_answer_text.lower():
        final_answer = 'yes'
    elif 'no' in majority_answer_text.lower():
        final_answer = 'no'
    else:
        final_answer = 'other' # Handle cases where the model's answer is not yes/no

    return confidence, final_answer

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates the Expected Calibration Error."""
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.
    for i in range(n_bins):
        in_bin = (binids == i)
        if np.mean(in_bin) > 0:
            diff = np.abs(np.mean(y_prob[in_bin]) - np.mean(y_true[in_bin]))
            ece += diff * np.mean(in_bin)
    return ece



def main():
    
    print("Loading Models and Data")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading LLM: {MODEL_NAME} (with 4-bit quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto", 
    )

    print("Loading Similarity Model...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print(f"Loading Dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="validation")
    sample_dataset = dataset.select(range(NUM_SAMPLES_FOR_DEMO))
    print(f"Using {NUM_SAMPLES_FOR_DEMO} samples from the dataset.")

    # Data Collection
    print("\nCollecting Raw Model Predictions")
    results_list = []
    for item in tqdm(sample_dataset, desc="Collecting model predictions"):
        prompt = f"Question: {item['question']}\nAnswer with only 'yes' or 'no':"
        raw_samples = get_mcdo_predictions(model, tokenizer, prompt, device, T=T_SAMPLES_PER_PROMPT)
        results_list.append({
            "question": item['question'],
            "ground_truth": "yes" if item['answer'] else "no",
            "raw_samples": raw_samples
        })

    # Confidence Scoring (Calculating P(true)) ---
    print("\nScoring Confidence of Each Prediction")
    scored_results = []
    for item in tqdm(results_list, desc="Scoring confidence (P(true))"):
        confidence, final_answer = calculate_confidence_and_answer(item['raw_samples'], similarity_model, device)
        is_correct = 1 if final_answer == item['ground_truth'] else 0
        scored_results.append({
            "question": item['question'],
            "ground_truth": item['ground_truth'],
            "final_answer": final_answer,
            "confidence_score": confidence, # This is our P(true) estimate
            "is_correct": is_correct
        })
    df_final = pd.DataFrame(scored_results)
    print("\nScoring complete. Sample of final data:")
    print(df_final[['question', 'final_answer', 'confidence_score', 'is_correct']].head())

    # Evaluate the P(true) Estimation
    print("\nEvaluating the P(true) Estimation")
    
    if df_final.empty or len(df_final['is_correct'].unique()) < 2:
        print("\nNot enough data or only one class present. Skipping evaluation plots.")
        return

    y_true = df_final['is_correct']
    y_scores = df_final['confidence_score']

    # AUROC Evaluation
    auroc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC Curve for Confidence Score Discrimination')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("auroc_curve.png")
    print(f"AUROC Score: {auroc_score:.4f}")

    # ECE and Reliability Plot Evaluation
    ece_score = calculate_ece(y_true.values, y_scores.values, n_bins=N_BINS_FOR_ECE)
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=N_BINS_FOR_ECE, strategy='uniform')

    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Llama Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Confidence (P(true) Estimate)')
    plt.ylabel('Fraction of Positives (Actual Accuracy)')
    plt.title(f'Reliability Plot (ECE = {ece_score:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig("reliability_plot.png")
    print(f"Expected Calibration Error (ECE): {ece_score:.4f}")

    # Confidence Distribution Plot
    plt.figure(figsize=(12, 7))
    sns.histplot(df_final[df_final['is_correct'] == 1], x='confidence_score', color="blue", label='Correct Answers', kde=True, stat="density", element="step")
    sns.histplot(df_final[df_final['is_correct'] == 0], x='confidence_score', color="red", label='Incorrect Answers', kde=True, stat="density", element="step")
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score (P(true) Estimate)')
    plt.legend()
    plt.savefig("confidence_distribution.png")

    print("\nEvaluation complete. Plots saved as .png files and displayed below.")
    plt.show() 


if __name__ == "__main__":
    main()