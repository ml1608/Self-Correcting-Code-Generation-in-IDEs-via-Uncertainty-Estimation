# Pearson Correlation Computation for Probes

This folder contains scripts to compute Pearson correlation scores between probe predictions and semantic entropy for test set examples.

## Folder Structure

This folder is self-contained and includes all necessary files:

```
PearsonProbes/
├── compute_pearson_scores.py  # Main script
├── saved_probes/              # All trained probes (6 probe directories)
│   ├── meta-llama_Llama-3.2-3B-Instruct_SLT_mlp/
│   ├── meta-llama_Llama-3.2-3B-Instruct_TBG_mlp/
│   ├── Qwen_Qwen2.5-Coder-3B-Instruct_SLT_mlp/
│   ├── Qwen_Qwen2.5-Coder-3B-Instruct_TBG_mlp/
│   ├── deepseek-ai_deepseek-coder-1.3b-instruct_SLT_mlp/
│   └── deepseek-ai_deepseek-coder-1.3b-instruct_TBG_mlp/
├── DatasetSplit/              # Dataset splits
│   └── test_tasks.csv         # Test task IDs
└── README.md
```

Each probe directory contains:
- `probe.pkl` - The trained probe (scaler + classifier)
- `probe_metadata.json` - Probe metadata

## Environment Variables (Optional)

You can override default paths using environment variables:

- `SAVED_PROBES_DIR` - Path to saved_probes directory (default: `./saved_probes`)
- `DATASET_SPLIT_DIR` - Path to DatasetSplit directory (default: `./DatasetSplit`)
- `OUTPUT_DIR` - Path for output files (default: current directory)
- `HF_TOKEN` - Hugging Face token (required)

## Usage

```bash
python compute_pearson_scores.py
```

## Output

The script generates:
- `pearson_scores.csv` - CSV file with Pearson correlation results
- `pearson_scores.json` - JSON file with the same results

Each row contains:
- `model_id` - Model identifier
- `model_family` - Model family (llama, qwen-coder-instruct, deepseek)
- `feature_method` - Feature extraction method (SLT or TBG)
- `probe_type` - Probe type (SLT+MLP or TBG+MLP)
- `pearson_r` - Pearson correlation coefficient
- `pearson_p` - P-value
- `n_samples` - Number of test examples used

## Note

Since the probe dataset CSV files have truncated arrays, the script automatically regenerates features for test set tasks. This requires loading the models, which may take some time.

