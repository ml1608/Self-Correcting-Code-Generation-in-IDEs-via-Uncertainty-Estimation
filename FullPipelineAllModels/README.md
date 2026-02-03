# Full Pipeline for All Models

This folder contains the complete pipeline implementation that evaluates all models (Llama, Qwen, DeepSeek) with both SLT and TBG feature methods.

## Overview

The pipeline:
1. **Uses new probes** from `Dataset+Probes/saved_probes/` (TBG+MLP and SLT+MLP for all 3 models)
2. **Uses optimized thresholds** from `threshold_recommendations.csv` (best thresholds from threshold analysis)
3. **Uses dataset splits** from `Dataset+Probes/DatasetSplit/` (70/15/15 train/val/test split)
4. **Evaluates all combinations**: 3 models × 2 feature methods = 6 total evaluations

## Files

- `run_pipeline_all_models.py`: Main script that runs the pipeline for all models and feature methods
- `adaptive_decoding_lambda.py`: Adaptive decoding implementation (supports both SLT and TBG)
- `self_correction_lambda.py`: Self-correction implementation (supports both SLT and TBG)
- `sep_training_lambda.py`: Probe training (not used in this pipeline, probes are pre-trained)

## Key Features

### Feature Extraction
- **SLT (Second-to-Last Token)**: Extracts features from the second-to-last token position
- **TBG (Token Before Generation)**: Extracts features from the token before generation starts (prompt_len - 1)

### Probe Loading
- Probes are loaded from `../Dataset+Probes/saved_probes/{model_name}_{feature_method}_mlp/`
- Thresholds are loaded from `../Dataset+Probes/threshold_analysis_plots/threshold_recommendations.csv`
- Each probe includes: scaler, classifier, threshold, and feature_method

### Dataset Splits
- Test set tasks are filtered using `../Dataset+Probes/DatasetSplit/test_tasks.csv`
- This ensures evaluation uses the same test set as probe training

## Usage

### Run Full Pipeline

```bash
python run_pipeline_all_models.py
```

This will:
1. Load all probes and thresholds
2. Filter tasks to test set
3. Run baseline, adaptive decoding, and self-correction for each model/feature combination
4. Save results to `pipeline_results_all_models/`

### Models Evaluated

1. **meta-llama/Llama-3.2-3B-Instruct** (SLT and TBG)
2. **Qwen/Qwen2.5-Coder-3B-Instruct** (SLT and TBG)
3. **deepseek-ai/deepseek-coder-1.3b-instruct** (SLT and TBG)

### Output

Results are saved as JSON files in `pipeline_results_all_models/`:
- `all_models_results_{timestamp}.json`: Combined results for all model/feature combinations

Each result includes:
- Baseline Pass@1
- Adaptive Decoding Pass@1 and improvement
- Self-Correction Pass@1 and improvement
- Latency comparisons
- Task-level results

## Requirements

- Hugging Face token with access to gated models (Llama, Qwen, DeepSeek)
- All probes must be trained and saved in `Dataset+Probes/saved_probes/`
- Threshold recommendations CSV must exist
- Dataset splits must exist in `Dataset+Probes/DatasetSplit/`

## Differences from Original Pipeline

1. **Multiple Models**: Evaluates all 3 models instead of just one
2. **Multiple Feature Methods**: Evaluates both SLT and TBG instead of just SLT
3. **Optimized Thresholds**: Uses best thresholds from threshold analysis instead of median
4. **Test Set Only**: Evaluates only on test set tasks (matching probe training split)
5. **No Probe Training**: Assumes probes are already trained (loads from saved_probes/)

## Notes

- The pipeline loads models sequentially (one at a time) to manage GPU memory
- Each model is unloaded after evaluation to free GPU memory
- Results are saved incrementally, so partial results are preserved if the pipeline is interrupted

