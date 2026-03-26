#!/usr/bin/env python3
"""
Analysis Script for Pearson Correlation Results

This script analyzes the Pearson correlation scores between probe predictions
and semantic entropy, providing detailed explanations and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================
# Load Data
# ============================================================

SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "pearson_scores.csv"
JSON_FILE = SCRIPT_DIR / "pearson_scores.json"

print("="*80)
print("PEARSON CORRELATION ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv(CSV_FILE)
print(f"\n✅ Loaded {len(df)} probe results from {CSV_FILE}")

# ============================================================
# Understanding Pearson Correlation
# ============================================================

print("\n" + "="*80)
print("UNDERSTANDING PEARSON CORRELATION")
print("="*80)

print("""
PEARSON r (Correlation Coefficient):
  - Range: -1 to +1
  - +1: Perfect positive correlation (as one increases, the other increases proportionally)
  - 0: No linear correlation
  - -1: Perfect negative correlation (as one increases, the other decreases proportionally)
  
  Interpretation:
    |r| > 0.7: Strong correlation
    0.5 < |r| < 0.7: Moderate correlation
    0.3 < |r| < 0.5: Weak correlation
    |r| < 0.3: Very weak or no correlation

PEARSON p (P-value):
  - Range: 0 to 1
  - Probability that the observed correlation occurred by chance
  - Lower p-value = stronger evidence of a real correlation
  
  Significance levels:
    p < 0.001: Highly significant (***)
    p < 0.01: Very significant (**)
    p < 0.05: Significant (*)
    p < 0.1: Marginally significant (.)
    p >= 0.1: Not significant

WHAT THIS MEANS FOR YOUR PROBES:
  - High positive r: Probe predictions strongly correlate with semantic entropy
  - Low p-value: The correlation is statistically significant (not due to chance)
  - This indicates the probe is successfully capturing uncertainty information
""")

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nOverall Statistics:")
print(f"  Mean Pearson r: {df['pearson_r'].mean():.4f}")
print(f"  Median Pearson r: {df['pearson_r'].median():.4f}")
print(f"  Std Dev Pearson r: {df['pearson_r'].std():.4f}")
print(f"  Min Pearson r: {df['pearson_r'].min():.4f}")
print(f"  Max Pearson r: {df['pearson_r'].max():.4f}")

print(f"\nBy Feature Method:")
for method in df['feature_method'].unique():
    method_df = df[df['feature_method'] == method]
    print(f"\n  {method}:")
    print(f"    Mean r: {method_df['pearson_r'].mean():.4f}")
    print(f"    Mean p: {method_df['pearson_p'].mean():.6f}")
    print(f"    Significant (p<0.05): {sum(method_df['pearson_p'] < 0.05)}/{len(method_df)}")

print(f"\nBy Model Family:")
for family in df['model_family'].unique():
    family_df = df[df['model_family'] == family]
    print(f"\n  {family}:")
    print(f"    Mean r: {family_df['pearson_r'].mean():.4f}")
    print(f"    Mean p: {family_df['pearson_p'].mean():.6f}")

# ============================================================
# Detailed Results Table
# ============================================================

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)

# Add interpretation columns
def interpret_r(r):
    """Interpret Pearson r value."""
    abs_r = abs(r)
    if abs_r > 0.7:
        return "Strong"
    elif abs_r > 0.5:
        return "Moderate"
    elif abs_r > 0.3:
        return "Weak"
    else:
        return "Very Weak"

def interpret_p(p):
    """Interpret p-value."""
    if p < 0.001:
        return "*** (Highly Significant)"
    elif p < 0.01:
        return "** (Very Significant)"
    elif p < 0.05:
        return "* (Significant)"
    elif p < 0.1:
        return ". (Marginal)"
    else:
        return "Not Significant"

df['r_interpretation'] = df['pearson_r'].apply(interpret_r)
df['p_interpretation'] = df['pearson_p'].apply(interpret_p)

# Display results
print("\nDetailed Results:")
print("-" * 80)
for idx, row in df.iterrows():
    print(f"\n{row['model_family'].upper()} - {row['feature_method']}+MLP:")
    print(f"  Model: {row['model_id']}")
    print(f"  Pearson r: {row['pearson_r']:.4f} ({row['r_interpretation']})")
    print(f"  P-value: {row['pearson_p']:.6f} {row['p_interpretation']}")
    print(f"  N samples: {row['n_samples']}")
    
    # Interpretation
    if row['pearson_r'] > 0.7 and row['pearson_p'] < 0.05:
        print(f"  ✓ Excellent: Strong, significant correlation")
    elif row['pearson_r'] > 0.5 and row['pearson_p'] < 0.05:
        print(f"  ✓ Good: Moderate, significant correlation")
    elif row['pearson_r'] > 0.3 and row['pearson_p'] < 0.05:
        print(f"  ⚠ Fair: Weak but significant correlation")
    elif row['pearson_p'] >= 0.05:
        print(f"  ✗ Poor: Correlation not statistically significant")

# ============================================================
# Visualizations
# ============================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create output directory for plots
plots_dir = SCRIPT_DIR / "analysis_plots"
plots_dir.mkdir(exist_ok=True)

# 1. Bar plot of Pearson r values
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df.sort_values('pearson_r', ascending=True)
colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' 
          for p in df_sorted['pearson_p']]
bars = ax.barh(range(len(df_sorted)), df_sorted['pearson_r'], color=colors)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels([f"{row['model_family']}-{row['feature_method']}" 
                    for _, row in df_sorted.iterrows()])
ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12)
ax.set_title('Pearson Correlation by Probe\n(Green=Significant, Orange=Marginal, Red=Not Significant)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0.7, color='blue', linestyle='--', alpha=0.5, label='Strong (r>0.7)')
ax.axvline(x=0.5, color='purple', linestyle='--', alpha=0.5, label='Moderate (r>0.5)')
ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='Weak (r>0.3)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    ax.text(row['pearson_r'] + 0.02, i, f"{row['pearson_r']:.3f}", 
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(plots_dir / 'pearson_r_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_dir / 'pearson_r_comparison.png'}")
plt.close()

# 2. Comparison: SLT vs TBG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# By feature method
slt_df = df[df['feature_method'] == 'SLT']
tbg_df = df[df['feature_method'] == 'TBG']

ax1.bar(['SLT', 'TBG'], 
        [slt_df['pearson_r'].mean(), tbg_df['pearson_r'].mean()],
        color=['steelblue', 'coral'], alpha=0.7)
ax1.errorbar(['SLT', 'TBG'],
             [slt_df['pearson_r'].mean(), tbg_df['pearson_r'].mean()],
             yerr=[slt_df['pearson_r'].std(), tbg_df['pearson_r'].std()],
             fmt='none', color='black', capsize=5)
ax1.set_ylabel('Mean Pearson r', fontsize=12)
ax1.set_title('Mean Correlation by Feature Method', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1])

# By model
model_order = ['llama', 'qwen-coder-instruct', 'deepseek']
model_labels = ['Llama', 'Qwen', 'DeepSeek']
x_pos = np.arange(len(model_order))
width = 0.35

slt_means = [df[(df['model_family'] == m) & (df['feature_method'] == 'SLT')]['pearson_r'].values[0] 
             if len(df[(df['model_family'] == m) & (df['feature_method'] == 'SLT')]) > 0 else 0
             for m in model_order]
tbg_means = [df[(df['model_family'] == m) & (df['feature_method'] == 'TBG')]['pearson_r'].values[0] 
             if len(df[(df['model_family'] == m) & (df['feature_method'] == 'TBG')]) > 0 else 0
             for m in model_order]

ax2.bar(x_pos - width/2, slt_means, width, label='SLT', color='steelblue', alpha=0.7)
ax2.bar(x_pos + width/2, tbg_means, width, label='TBG', color='coral', alpha=0.7)
ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('Pearson r', fontsize=12)
ax2.set_title('Correlation by Model and Feature Method', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_labels)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(plots_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_dir / 'method_comparison.png'}")
plt.close()

# 3. P-value visualization
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df.sort_values('pearson_p', ascending=True)
colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' 
          for p in df_sorted['pearson_p']]
bars = ax.barh(range(len(df_sorted)), -np.log10(df_sorted['pearson_p']), color=colors)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels([f"{row['model_family']}-{row['feature_method']}" 
                    for _, row in df_sorted.iterrows()])
ax.set_xlabel('-log10(p-value)', fontsize=12)
ax.set_title('Statistical Significance (-log10 p-value)\n(Green=Significant, Orange=Marginal, Red=Not Significant)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=-np.log10(0.05), color='blue', linestyle='--', alpha=0.5, label='p=0.05')
ax.axvline(x=-np.log10(0.01), color='purple', linestyle='--', alpha=0.5, label='p=0.01')
ax.axvline(x=-np.log10(0.001), color='red', linestyle='--', alpha=0.5, label='p=0.001')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    ax.text(-np.log10(row['pearson_p']) + 0.2, i, f"p={row['pearson_p']:.4f}", 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(plots_dir / 'pvalue_significance.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_dir / 'pvalue_significance.png'}")
plt.close()

# 4. Summary table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['model_family'],
        row['feature_method'],
        f"{row['pearson_r']:.4f}",
        f"{row['pearson_p']:.6f}",
        row['p_interpretation'].split('(')[0].strip()
    ])

table = ax.table(cellText=table_data,
                colLabels=['Model', 'Method', 'Pearson r', 'P-value', 'Significance'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code cells based on significance
for i in range(1, len(table_data) + 1):
    p_val = float(table_data[i-1][3])
    if p_val < 0.05:
        table[(i, 4)].set_facecolor('#90EE90')  # Light green
    elif p_val < 0.1:
        table[(i, 4)].set_facecolor('#FFE4B5')  # Moccasin
    else:
        table[(i, 4)].set_facecolor('#FFB6C1')  # Light pink

plt.title('Pearson Correlation Results Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(plots_dir / 'results_summary_table.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_dir / 'results_summary_table.png'}")
plt.close()

# ============================================================
# Key Findings
# ============================================================

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

best_probe = df.loc[df['pearson_r'].idxmax()]
worst_probe = df.loc[df['pearson_r'].idxmin()]

print(f"\n🏆 Best Performing Probe:")
print(f"   {best_probe['model_family']} - {best_probe['feature_method']}+MLP")
print(f"   Pearson r: {best_probe['pearson_r']:.4f}")
print(f"   P-value: {best_probe['pearson_p']:.6f}")

print(f"\n⚠️  Weakest Performing Probe:")
print(f"   {worst_probe['model_family']} - {worst_probe['feature_method']}+MLP")
print(f"   Pearson r: {worst_probe['pearson_r']:.4f}")
print(f"   P-value: {worst_probe['pearson_p']:.6f}")

# SLT vs TBG comparison
slt_mean = slt_df['pearson_r'].mean()
tbg_mean = tbg_df['pearson_r'].mean()
print(f"\n📊 Feature Method Comparison:")
print(f"   SLT mean r: {slt_mean:.4f}")
print(f"   TBG mean r: {tbg_mean:.4f}")
if slt_mean > tbg_mean:
    print(f"   → SLT performs {((slt_mean/tbg_mean - 1) * 100):.1f}% better on average")
else:
    print(f"   → TBG performs {((tbg_mean/slt_mean - 1) * 100):.1f}% better on average")

# Significance summary
significant = sum(df['pearson_p'] < 0.05)
marginal = sum((df['pearson_p'] >= 0.05) & (df['pearson_p'] < 0.1))
not_sig = sum(df['pearson_p'] >= 0.1)

print(f"\n📈 Statistical Significance:")
print(f"   Highly significant (p<0.001): {sum(df['pearson_p'] < 0.001)}/{len(df)}")
print(f"   Significant (p<0.05): {significant}/{len(df)}")
print(f"   Marginal (0.05≤p<0.1): {marginal}/{len(df)}")
print(f"   Not significant (p≥0.1): {not_sig}/{len(df)}")

print("\n" + "="*80)
print("✅ Analysis complete! Check the 'analysis_plots' folder for visualizations.")
print("="*80)


