# Full Pipeline

This folder contains the complete evaluation pipeline for Semantic Entropy Probe (SEP)-guided self-correction and adaptive decoding. It supports any HuggingFace dataset and any causal language model — not just the three default models or BigCodeBench.

## Overview

The pipeline:
1. Loads pre-trained SEP probes from `Dataset+Probes/saved_probes/<dataset>/`
2. Loads F1-optimized thresholds from `Dataset+Probes/threshold_analysis_plots/<dataset>/threshold_recommendations.csv`
3. Filters tasks to the held-out test split from `Dataset+Probes/DatasetSplit/<dataset>/`
4. Runs two distinct self-correction approaches:
   - **Baseline**: greedy decoding (temp=0.0)
   - **Full Function Regeneration (SLT only)**: resamples up to 5 times when uncertainty is high
   - **Proactive Regeneration (TBG only)**: triggers AdaDec generation when uncertainty is high

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
| `adaptive_decoding_lambda.py` | Legacy custom adaptive decoding path (kept for reference) |
| `adaptive_decoding_adadec.py` | AdaDec-backed proactive regeneration: delegates generation to AdaDec's `Generator` (AdaFixL mode) after SEP task-level gating |
| `self_correction_lambda.py` | MTE-style iterative resampling using SLT probe |
| `utils.py` | Shared helpers: feature extraction, uncertainty estimation, path resolution, model family inference |
| `sep_training_lambda.py` | Standalone probe training script (not used by the pipeline — probes are pre-trained via `Dataset+Probes/train_probes.py`) |
| `ada_dec_thresholds.json` | Per-model token entropy thresholds for adaptive decoding fallback |
| `AdaDec/` | Git submodule — original AdaDec implementation (see below) |

### AdaDec

`AdaDec/` contains the original AdaDec implementation ([paper](https://arxiv.org/abs/2406.12399), [repo](https://github.com/SYSUSELab/AdaDec)). The key file used by this pipeline is `AdaDec/llm/generator.py`, which provides the `Generator` class.

AdaDec's `Generator` operates in **AdaFixL** mode: at each generation step it computes the Shannon entropy of the next-token distribution. If entropy exceeds a learned threshold it uses lookahead beam reranking; otherwise it uses greedy selection. This is a purely token-level decision, orthogonal to the task-level gating done by the SEP probe.

The main runner is already configured to use `adaptive_decoding_adadec.py` for proactive regeneration.

### Strategy split enforced by the runner

- **SLT probes**: run full-function self-correction only (uncertainty-based and verification-based).
- **TBG probes**: run proactive regeneration only (AdaDec generation with token-level entropy gating).

This mirrors the intended design where TBG serves as an early task-level gate and AdaDec handles token-level correction dynamics.

### AdaDec threshold learning (logistic-regression learned)

AdaDec's token-level threshold is external to SEP probe training. To learn/update it per dataset:

1. Add your dataset support in `AdaDec/src/learn_threshold/generate_data.py`.
2. Follow steps 1 and 2 in the AdaDec README:
   - [https://github.com/SYSUSELab/AdaDec/tree/main](https://github.com/SYSUSELab/AdaDec/tree/main)
3. Export learned thresholds to JSON and pass them to the runner:

```bash
python run_pipeline_all_models.py \
  --adadec_thresholds_json /path/to/adadec_thresholds.json
```

Supported threshold JSON formats:
- Flat: `{ "llama3.2-3b": 0.31, "qwen2.5-coder-3b": 0.28, ... }`
- Dataset-aware: `{ "bigcodebench": { ... }, "mbpp": { ... } }`

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
  --classifiers linreg
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
  --classifiers linreg \
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
| `--classifiers` | `mlp logreg` | Probe classifier types (`linreg` supported and recommended for current setup) |
| `--limit_tasks` | None (all) | Cap the number of test tasks |
| `--output_dir` | `pipeline_results_all_models` | Directory for output JSON files |
| `--adadec_thresholds_json` | None | Optional AdaDec token-threshold JSON (flat or dataset-aware format) |

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

- **SLT full-function regeneration**: SLT uncertainty is computed from generated completions and drives iterative resampling (`temp=0.0` first attempt, then `temp=0.3`) until uncertainty falls below threshold or tests pass (verification mode).
- **TBG proactive regeneration**: TBG uncertainty is computed before full generation and only decides whether to trigger AdaDec. Once triggered, AdaDec applies token-level entropy gating and lookahead reranking.
- **Model family inference**: The model family (used for chat template formatting) is inferred automatically from the model ID. For unknown architectures, the tokenizer's built-in chat template is used as a fallback.
- **Artifact path derivation**: Paths to probes, thresholds, and splits are derived from the dataset name automatically, matching the layout produced by `train_probes.py` and `thresh_tune.py`.
- **GPU memory**: Models are loaded and unloaded sequentially to fit within a single GPU.
- **Partial results**: Results are saved after all combinations complete, but each model's results are appended as they finish so a crash mid-run preserves partial output.
