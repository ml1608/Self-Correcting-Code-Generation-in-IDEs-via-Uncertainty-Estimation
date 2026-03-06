# Full Pipeline

This folder contains the complete evaluation pipeline for Semantic Entropy Probe (SEP)-guided self-correction and adaptive decoding. It supports any HuggingFace dataset and any causal language model — not just the three default models or BigCodeBench.

## Overview

The pipeline:
1. Loads pre-trained SEP probes from `Dataset+Probes/saved_probes/<dataset>/`
2. Loads F1-optimized thresholds from `Dataset+Probes/threshold_analysis_plots/<dataset>/threshold_recommendations.csv`
3. Filters tasks to the held-out test split from `Dataset+Probes/DatasetSplit/<dataset>/`
4. Runs three evaluations per model × feature method × classifier combination:
   - **Baseline**: greedy decoding (temp=0.0)
   - **Adaptive Decoding**: switches to beam search with lookahead when the probe predicts high entropy
   - **Self-Correction**: resamples up to 5 times when the probe predicts high entropy (SLT only)

Artifact paths (probes, thresholds, splits) are derived automatically from the dataset name. Model family is inferred automatically from the model ID.

## Git Submodules

This folder includes the [AdaDec](https://github.com/SYSUSELab/AdaDec) repository as a git submodule under `AdaDec/`. After cloning the repo, initialize it with:

```bash
git submodule update --init --recursive
```

If you already have the repo cloned and the `AdaDec/` directory is empty, run the same command. Without this step, `adaptive_decoding_adadec.py` will fall back with a warning and the AdaDec-backed implementation will be unavailable.

## Files

| File | Description |
|---|---|
| `run_pipeline_all_models.py` | Main entrypoint — accepts CLI arguments for dataset, models, etc. |
| `adaptive_decoding_lambda.py` | Custom adaptive decoding: per-token probe gating with a hand-rolled lookahead loop |
| `adaptive_decoding_adadec.py` | AdaDec-backed adaptive decoding: delegates generation to AdaDec's `Generator` (AdaFixL mode); same SEP probe task-level gating as `adaptive_decoding_lambda.py` |
| `self_correction_lambda.py` | MTE-style iterative resampling using SLT probe |
| `utils.py` | Shared helpers: feature extraction, uncertainty estimation, path resolution, model family inference |
| `sep_training_lambda.py` | Standalone probe training script (not used by the pipeline — probes are pre-trained via `Dataset+Probes/train_probes.py`) |
| `ada_dec_thresholds.json` | Per-model token entropy thresholds for adaptive decoding fallback |
| `AdaDec/` | Git submodule — original AdaDec implementation (see below) |

### AdaDec

`AdaDec/` contains the original AdaDec implementation ([paper](https://arxiv.org/abs/2406.12399), [repo](https://github.com/SYSUSELab/AdaDec)). The key file used by this pipeline is `AdaDec/llm/generator.py`, which provides the `Generator` class.

AdaDec's `Generator` operates in **AdaFixL** mode: at each generation step it computes the Shannon entropy of the next-token distribution. If entropy exceeds a learned threshold it uses lookahead beam reranking; otherwise it uses greedy selection. This is a purely token-level decision, orthogonal to the task-level gating done by the SEP probe.

**Switching between adaptive decoding implementations**: Both `adaptive_decoding_lambda.py` and `adaptive_decoding_adadec.py` expose the same public API (`adaptive_decode`, `evaluate_adaptive_decoding`). To switch implementations in `run_pipeline_all_models.py`, change the import:

```python
# Original custom implementation (default)
from adaptive_decoding_lambda import adaptive_decode, evaluate_adaptive_decoding

# AdaDec-backed implementation
from adaptive_decoding_adadec import adaptive_decode, evaluate_adaptive_decoding
```

## Usage

### Default run (BigCodeBench, 3 default models)

```bash
python run_pipeline_all_models.py
```

### Different dataset

```bash
python run_pipeline_all_models.py \
  --dataset openai_humaneval \
  --split test \
  --prompt_field prompt
```

### Custom models

```bash
python run_pipeline_all_models.py \
  --models mistralai/Mistral-7B-Instruct-v0.3 google/codegemma-7b-it
```

### Subset of feature methods or classifiers

```bash
python run_pipeline_all_models.py \
  --feature_methods SLT \
  --classifiers mlp
```

### Quick test run

```bash
python run_pipeline_all_models.py --limit_tasks 10
```

### Custom probes directory

```bash
python run_pipeline_all_models.py --probes_dir /path/to/my_probes
```

### Full example

```bash
python run_pipeline_all_models.py \
  --dataset bigcode/bigcodebench \
  --split v0.1.4 \
  --prompt_field instruct_prompt \
  --models meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct \
  --feature_methods SLT TBG \
  --classifiers mlp logreg \
  --limit_tasks 100 \
  --output_dir my_results
```

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--models` | Llama, Qwen, DeepSeek | HuggingFace model IDs to evaluate |
| `--dataset` | `bigcode/bigcodebench` | HuggingFace dataset name |
| `--split` | `v0.1.4` | Dataset split |
| `--prompt_field` | `instruct_prompt` | Field name containing the prompt |
| `--probes_dir` | Auto-derived from dataset | Path to saved probe subdirectories |
| `--feature_methods` | `SLT TBG` | Feature extraction methods |
| `--classifiers` | `mlp logreg` | Probe classifier types |
| `--limit_tasks` | None (all) | Cap the number of test tasks |
| `--output_dir` | `pipeline_results_all_models` | Directory for output JSON files |

If `--dataset`, `--split`, or `--prompt_field` are omitted, the pipeline reads them from `split_summary.json` in the split directory, ensuring consistency with the dataset used during probe training.

## Prerequisites

- A Hugging Face token (set via `HF_TOKEN` env var or entered at prompt) with access to any gated models
- Probes trained and saved under `Dataset+Probes/saved_probes/<dataset_safe_name>/`
- Threshold recommendations CSV at `Dataset+Probes/threshold_analysis_plots/<dataset_safe_name>/threshold_recommendations.csv`
- Dataset splits at `Dataset+Probes/DatasetSplit/<dataset_safe_name>/`

Run `Dataset+Probes/train_probes.py` and `Dataset+Probes/thresh_tune.py` first to generate these artifacts for a new dataset.

## Output

Results are saved to `pipeline_results_all_models/` (or `--output_dir`) as:

```
all_models_results_{timestamp}.json
```

Each entry in the JSON contains:
- `model_id`, `feature_method`, `classifier`, `threshold`
- `baseline_results`: Pass@1 and per-task results for greedy decoding
- `adaptive_results`: Pass@1 and improvement for adaptive decoding (TBG only)
- `uncertainty_correction_results`: Pass@1 and improvement for uncertainty-guided self-correction (SLT only)
- `verification_correction_results`: Pass@1 and improvement for verification-guided self-correction (SLT only)
- `comparison`: summary table across all methods

## Key Design Notes

- **TBG vs SLT for self-correction**: TBG features are extracted from the prompt before generation, so they are static per task. They cannot drive self-correction (the uncertainty score would not change between attempts). Self-correction is therefore only run with SLT probes.
- **Model family inference**: The model family (used for chat template formatting) is inferred automatically from the model ID. For unknown architectures, the tokenizer's built-in chat template is used as a fallback.
- **Artifact path derivation**: Paths to probes, thresholds, and splits are derived from the dataset name automatically, matching the layout produced by `train_probes.py` and `thresh_tune.py`.
- **GPU memory**: Models are loaded and unloaded sequentially to fit within a single GPU.
- **Partial results**: Results are saved after all combinations complete, but each model's results are appended as they finish so a crash mid-run preserves partial output.
