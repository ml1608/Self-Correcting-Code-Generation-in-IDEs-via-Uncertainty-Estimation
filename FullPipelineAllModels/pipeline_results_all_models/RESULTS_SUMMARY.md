# Full Pipeline Results Summary

**Date:** 2026-02-05 (Updated with DeepSeek fixes)  
**Test Set:** 25 tasks (from DatasetSplit)

## Results Table

| Model | Method | Baseline | Adaptive Decoding | Self-Correction |
|-------|--------|----------|-------------------|-----------------|
| **Llama-3.2-3B** | SLT | 0.360 (4.62s) | 0.480 (+0.120, 97.33s) | 0.440 (+0.080, 12.86s) |
| **Llama-3.2-3B** | TBG | 0.360 (4.73s) | 0.360 (+0.000, 30.90s) | **0.520 (+0.160, 8.39s)** |
| **Qwen2.5-Coder-3B** | SLT | 0.760 (6.37s) | 0.680 (-0.080, 137.36s) | 0.920 (+0.160, 18.19s) |
| **Qwen2.5-Coder-3B** | TBG | 0.760 (6.35s) | 0.680 (-0.080, 65.46s) | **0.960 (+0.200, 13.06s)** |
| **DeepSeek-Coder-1.3B** | SLT | **0.560 (4.83s)** | 0.480 (-0.080, 17.54s) | **0.640 (+0.080, 10.74s)** |
| **DeepSeek-Coder-1.3B** | TBG | **0.560 (4.96s)** | 0.560 (+0.000, 25.39s) | **0.640 (+0.080, 10.19s)** |

*Format: Pass@1 (Improvement, Latency)*

### DeepSeek Detailed Metrics (Updated 2026-02-05)

**DeepSeek-Coder-1.3B with SLT features:**
- **Baseline**: Pass@1: 0.5600, Latency: 4.827s
- **Adaptive Decoding**: Pass@1: 0.4800, Improvement: -0.0800 (-8.00%), Latency: 17.539s, Adaptive Ratio: 13.3%, Tasks Improved: 0
- **Self-Correction**: Pass@1: 0.6400, Improvement: +0.0800 (+8.00%), Latency: 10.737s, Avg Regenerations: 2.52, Tasks Improved: 3

**DeepSeek-Coder-1.3B with TBG features:**
- **Baseline**: Pass@1: 0.5600, Latency: 4.960s
- **Adaptive Decoding**: Pass@1: 0.5600, Improvement: +0.0000 (+0.00%), Latency: 25.387s, Adaptive Ratio: 24.3%, Tasks Improved: 0
- **Self-Correction**: Pass@1: 0.6400, Improvement: +0.0800 (+8.00%), Latency: 10.185s, Avg Regenerations: 2.24, Tasks Improved: 2

## Key Findings

### 1. Self-Correction Performance

**Best Performers:**
- **Qwen2.5-Coder-3B + TBG**: 0.960 Pass@1 (+0.200 improvement, 13.06s latency)
- **Llama-3.2-3B + TBG**: 0.520 Pass@1 (+0.160 improvement, 8.39s latency)
- **Qwen2.5-Coder-3B + SLT**: 0.920 Pass@1 (+0.160 improvement, 18.19s latency)

**Self-Correction consistently improves performance:**
- SLT: Average improvement of +0.107 (±0.040)
- TBG: Average improvement of +0.147 (±0.060)

### 2. Adaptive Decoding Performance

**Mixed Results:**
- **SLT**: Average improvement of +0.013 (±0.100) - highly variable
  - Llama: +0.120 improvement
  - Qwen: -0.080 degradation
  - DeepSeek: -0.080 degradation
- **TBG**: Average improvement of -0.027 (±0.046) - slight degradation
  - Llama: No change
  - Qwen: -0.080 degradation
  - DeepSeek: No change

**Adaptive Decoding has high latency:**
- SLT: Average 84.14s (±60.99s) - 15.7x slower than baseline
- TBG: Average 40.58s (±20.25s) - 7.6x slower than baseline

### 3. Latency Analysis

**Baseline Latency:**
- SLT: 5.35s ± 0.91s
- TBG: 5.35s ± 0.88s
- *Similar baseline latencies for both methods*

**Self-Correction Latency:**
- SLT: 13.93s ± 3.73s (2.6x slower than baseline)
- TBG: 11.15s ± 2.34s (2.1x slower than baseline)
- *TBG is faster for self-correction*

**Adaptive Decoding Latency:**
- SLT: 84.14s ± 60.99s (15.7x slower than baseline)
- TBG: 40.58s ± 20.25s (7.6x slower than baseline)
- *TBG is significantly faster for adaptive decoding*

### 4. Method Comparison: TBG vs SLT

| Metric | SLT | TBG | Winner |
|--------|-----|-----|--------|
| **Self-Correction Improvement** | +0.107 | +0.147 | TBG |
| **Self-Correction Latency** | 13.93s | 11.15s | TBG |
| **Adaptive Improvement** | +0.013 | -0.027 | SLT |
| **Adaptive Latency** | 84.14s | 40.58s | TBG |

**Overall Winner: TBG**
- Better self-correction performance (+0.147 vs +0.107)
- Faster self-correction (11.15s vs 13.93s)
- Much faster adaptive decoding (40.58s vs 84.14s)

### 5. Model-Specific Observations

**Llama-3.2-3B:**
- TBG self-correction performs best: 0.520 Pass@1 (+0.160)
- SLT adaptive decoding shows improvement (+0.120) but with very high latency (97.33s)

**Qwen2.5-Coder-3B:**
- Best overall performance with TBG self-correction: 0.960 Pass@1 (+0.200)
- Both methods show degradation with adaptive decoding
- Self-correction is highly effective for this model

**DeepSeek-Coder-1.3B:**
- **Fixed baseline performance**: Now shows 0.560 Pass@1 (was 0.000 due to code extraction issues)
- **Self-correction is effective**: Both SLT and TBG achieve 0.640 Pass@1 (+0.080 improvement)
- **Self-correction latency**: TBG is slightly faster (10.19s vs 10.74s)
- **Adaptive decoding**: SLT shows degradation (-0.080), TBG maintains baseline (0.000 improvement)
- **Key insight**: After fixing code extraction, DeepSeek performs comparably to other models with self-correction

## Recommendations

1. **Use TBG + Self-Correction** for best performance-to-latency trade-off
2. **Avoid Adaptive Decoding** for SLT on Qwen and DeepSeek (shows degradation)
3. **DeepSeek performance fixed**: After fixing code extraction issues, DeepSeek now shows competitive baseline (0.560) and strong self-correction results (0.640)
4. **Self-Correction is more reliable** than Adaptive Decoding across all models
5. **TBG is preferred** for DeepSeek self-correction (slightly faster: 10.19s vs 10.74s)

## Files Generated

- `pipeline_results_visualization.png`: Comprehensive visualization with 5 subplots
- `results_summary_table.csv`: Detailed results in CSV format
- `all_models_results_2026-02-04T07-31-17.743563.json`: Raw results data (initial run)
- `all_models_results_2026-02-05T03-31-55.471916.json`: Raw results data (updated with DeepSeek fixes)

