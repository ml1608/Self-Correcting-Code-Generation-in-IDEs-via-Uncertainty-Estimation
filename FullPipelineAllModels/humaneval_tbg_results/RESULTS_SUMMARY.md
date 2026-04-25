# HumanEval TBG Adaptive Decoding Results

**Dataset:** openai_humaneval | **Method:** TBG + AdaDec | **Models:** 3B scale

## Results Summary

| Model | Baseline Pass@1 | Adaptive Pass@1 | Improvement | Trigger Rate |
|-------|----------------|-----------------|-------------|--------------|
| Llama-3.2-3B-Instruct | 38.7% | 29.0% | +0.0% | 0.0% |
| Qwen2.5-Coder-3B-Instruct | 80.6% | 51.6% | +0.0% | 0.0% |
| deepseek-coder-1.3b-instruct | 0.0% | 16.1% | +0.0% | 0.0% |

## Notes
- Llama and Qwen TBG probes are poorly calibrated (predictions cluster near 0)
- AdaDec triggers too often for Llama/Qwen, overwriting correct greedy answers
- DeepSeek TBG probe is well calibrated (F1=0.70, threshold=1.04)
- DeepSeek baseline 0% is a known evaluation bug being investigated

## Thresholds Used (retuned on validation set)

| Model | Method | Threshold | F1 |
|-------|--------|-----------|-----|
| Llama-3.2-3B-Instruct | TBG | -0.0248 | - |
| Qwen2.5-Coder-3B-Instruct | TBG | -0.0311 | - |
| deepseek-coder-1.3b-instruct | TBG | 1.0403 | - |
## Corrected DeepSeek Results (baseline bug fixed)

| Model | Baseline Pass@1 | Adaptive Pass@1 | Improvement |
|-------|----------------|-----------------|-------------|
| deepseek-coder-1.3b | 61.3% | 54.8% | -6.5% (not significant) |

**Note:** Previous DeepSeek baseline of 0% was a bug — model was appending
# Test cases + print() statements which broke test execution. Fixed by
stripping those lines before evaluation.
