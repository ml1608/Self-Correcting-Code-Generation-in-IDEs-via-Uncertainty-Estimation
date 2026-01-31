#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Probe Experiment Results

This script analyzes the results from probe_experiment.py and provides:
- Summary statistics
- Best combinations
- Comparisons by feature method and classifier
- Visualizations (if matplotlib is available)
- Detailed probe metadata inspection
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  matplotlib/seaborn not available. Skipping visualizations.")

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(__file__).parent
CSV_FILE = RESULTS_DIR / "probe_experiment_results.csv"
JSON_FILE = RESULTS_DIR / "probe_experiment_results.json"
SAVED_PROBES_DIR = RESULTS_DIR / "saved_probes"

# ============================================================
# Load Data
# ============================================================

def load_results():
    """Load results from CSV and JSON files."""
    if not CSV_FILE.exists():
        print(f"❌ Results file not found: {CSV_FILE}")
        print("   Make sure you've run probe_experiment.py first.")
        return None, None
    
    df = pd.read_csv(CSV_FILE)
    
    results_json = None
    if JSON_FILE.exists():
        with open(JSON_FILE, 'r') as f:
            results_json = json.load(f)
    
    return df, results_json

# ============================================================
# Analysis Functions
# ============================================================

def print_summary(df):
    """Print overall summary statistics."""
    print("="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total combinations tested: {len(df)}")
    print(f"Feature methods: {', '.join(df['feature_method'].unique())}")
    print(f"Classifiers: {', '.join(df['classifier'].unique())}")
    print(f"\nAccuracy Statistics:")
    print(f"  Mean:   {df['test_accuracy'].mean():.4f}")
    print(f"  Std:    {df['test_accuracy'].std():.4f}")
    print(f"  Min:    {df['test_accuracy'].min():.4f}")
    print(f"  Max:    {df['test_accuracy'].max():.4f}")
    print(f"\nAUROC Statistics:")
    print(f"  Mean:   {df['test_auc'].mean():.4f}")
    print(f"  Std:    {df['test_auc'].std():.4f}")
    print(f"  Min:    {df['test_auc'].min():.4f}")
    print(f"  Max:    {df['test_auc'].max():.4f}")
    print("="*80 + "\n")

def print_all_results(df):
    """Print all results sorted by accuracy."""
    print("="*80)
    print("ALL RESULTS (Sorted by Test Accuracy)")
    print("="*80)
    df_sorted = df.sort_values('test_accuracy', ascending=False)
    
    # Format for better readability
    display_cols = ['feature_method', 'classifier', 'test_accuracy', 'test_auc', 
                   'feature_dim', 'n_train', 'n_val', 'n_test']
    print(df_sorted[display_cols].to_string(index=False))
    print("="*80 + "\n")

def print_all_9_combinations(df):
    """Print a clear 3x3 comparison table of all combinations."""
    print("="*80)
    print("ALL 9 COMBINATIONS: Feature Method × Classifier")
    print("="*80)
    print("\nThis table shows ALL combinations side-by-side for easy comparison:\n")
    
    # Create a pivot table for accuracy
    pivot_acc = df.pivot_table(
        values='test_accuracy',
        index='feature_method',
        columns='classifier',
        aggfunc='first'
    )
    
    # Create a pivot table for AUROC
    pivot_auc = df.pivot_table(
        values='test_auc',
        index='feature_method',
        columns='classifier',
        aggfunc='first'
    )
    
    # Reorder columns and rows for consistency
    feature_order = ['SLT', 'TBG', 'LAST']
    classifier_order = ['random_forest', 'mlp', 'deep_nn']
    
    pivot_acc = pivot_acc.reindex(index=feature_order, columns=classifier_order)
    pivot_auc = pivot_auc.reindex(index=feature_order, columns=classifier_order)
    
    print("TEST ACCURACY (Higher is better):")
    print("-" * 80)
    print("Feature Method | Random Forest |      MLP      |  Deep NN    |")
    print("-" * 80)
    for method in feature_order:
        if method in pivot_acc.index:
            row = f"{method:13s} |"
            for clf in classifier_order:
                if clf in pivot_acc.columns:
                    val = pivot_acc.loc[method, clf]
                    if pd.notna(val):
                        row += f"    {val:.4f}    |"
                    else:
                        row += f"     N/A      |"
                else:
                    row += f"     N/A      |"
            print(row)
    print("-" * 80)
    
    print("\nTEST AUROC (Higher is better, max=1.0):")
    print("-" * 80)
    print("Feature Method | Random Forest |      MLP      |  Deep NN    |")
    print("-" * 80)
    for method in feature_order:
        if method in pivot_auc.index:
            row = f"{method:13s} |"
            for clf in classifier_order:
                if clf in pivot_auc.columns:
                    val = pivot_auc.loc[method, clf]
                    if pd.notna(val):
                        row += f"    {val:.4f}    |"
                    else:
                        row += f"     N/A      |"
                else:
                    row += f"     N/A      |"
            print(row)
    print("-" * 80)
    
    # Find and highlight best combination
    best_idx = df['test_accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 BEST COMBINATION: {best['feature_method']} + {classifier_display_name(best['classifier'])}")
    print(f"   Accuracy: {best['test_accuracy']:.4f} | AUROC: {best['test_auc']:.4f}")
    
    # Rank all 9 combinations
    print("\n📊 RANKING (All 9 combinations sorted by accuracy):")
    print("-" * 80)
    df_ranked = df.sort_values('test_accuracy', ascending=False).reset_index(drop=True)
    for idx, row in df_ranked.iterrows():
        rank_emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{idx+1}."
        print(f"{rank_emoji:4s} {row['feature_method']:4s} + {classifier_display_name(row['classifier']):25s} | "
              f"Accuracy: {row['test_accuracy']:.4f} | AUROC: {row['test_auc']:.4f}")
    
    print("="*80 + "\n")

def print_best_combination(df):
    """Print details of the best performing combination."""
    print("="*80)
    print("🏆 BEST COMBINATION")
    print("="*80)
    
    best_idx = df['test_accuracy'].idxmax()
    best = df.loc[best_idx]
    
    print(f"Feature Method:     {best['feature_method']}")
    print(f"Classifier:          {best['classifier']}")
    print(f"Test Accuracy:      {best['test_accuracy']:.4f} ({best['test_accuracy']*100:.2f}%)")
    print(f"Test AUROC:         {best['test_auc']:.4f}")
    print(f"Feature Dimensions: {best['feature_dim']}")
    print(f"Dataset Split:      Train={best['n_train']}, Val={best['n_val']}, Test={best['n_test']}")
    if 'semantic_entropy_threshold' in best:
        print(f"Semantic Entropy Threshold: {best['semantic_entropy_threshold']:.4f}")
    if 'recommended_threshold' in best:
        print(f"Recommended Threshold:     {best['recommended_threshold']:.4f}")
    
    if 'probe_path' in best:
        print(f"Probe Path:         {best['probe_path']}")
    
    print("="*80 + "\n")

def print_comparison_by_feature_method(df):
    """Compare results grouped by feature extraction method."""
    print("="*80)
    print("COMPARISON BY FEATURE METHOD")
    print("="*80)
    
    for method in sorted(df['feature_method'].unique()):
        method_df = df[df['feature_method'] == method]
        best_method = method_df.loc[method_df['test_accuracy'].idxmax()]
        
        print(f"\n{method} (Second-to-Last Token)" if method == "SLT" else 
              f"\n{method} (Token Before Generation)" if method == "TBG" else
              f"\n{method} (Last Token - Baseline)" if method == "LAST" else
              f"\n{method}:")
        print(f"  Best classifier:        {best_method['classifier']}")
        print(f"  Best accuracy:         {best_method['test_accuracy']:.4f} ({best_method['test_accuracy']*100:.2f}%)")
        print(f"  Best AUROC:            {best_method['test_auc']:.4f}")
        print(f"  Mean accuracy (all):   {method_df['test_accuracy'].mean():.4f}")
        print(f"  Std accuracy:         {method_df['test_accuracy'].std():.4f}")
        print(f"  All classifiers:      {', '.join(method_df['classifier'].unique())}")
    
    print("="*80 + "\n")

def print_comparison_by_classifier(df):
    """Compare results grouped by classifier type."""
    print("="*80)
    print("COMPARISON BY CLASSIFIER")
    print("="*80)
    
    for clf in sorted(df['classifier'].unique()):
        clf_df = df[df['classifier'] == clf]
        best_clf = clf_df.loc[clf_df['test_accuracy'].idxmax()]
        
        print(f"\n{classifier_display_name(clf)}:")
        print(f"  Best feature method:   {best_clf['feature_method']}")
        print(f"  Best accuracy:        {best_clf['test_accuracy']:.4f} ({best_clf['test_accuracy']*100:.2f}%)")
        print(f"  Best AUROC:           {best_clf['test_auc']:.4f}")
        print(f"  Mean accuracy (all):  {clf_df['test_accuracy'].mean():.4f}")
        print(f"  Std accuracy:        {clf_df['test_accuracy'].std():.4f}")
        print(f"  All feature methods:  {', '.join(clf_df['feature_method'].unique())}")
    
    print("="*80 + "\n")

def classifier_display_name(clf):
    """Get display name for classifier."""
    names = {
        'random_forest': 'Random Forest',
        'mlp': 'MLP (sklearn)',
        'deep_nn': 'Deep Neural Network (PyTorch)'
    }
    return names.get(clf, clf)

def print_statistical_significance(df):
    """Print statistical comparisons."""
    print("="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)
    
    # Compare feature methods
    print("\nFeature Method Comparison:")
    for method in sorted(df['feature_method'].unique()):
        method_acc = df[df['feature_method'] == method]['test_accuracy'].values
        print(f"  {method}: mean={method_acc.mean():.4f}, std={method_acc.std():.4f}, n={len(method_acc)}")
    
    # Compare classifiers
    print("\nClassifier Comparison:")
    for clf in sorted(df['classifier'].unique()):
        clf_acc = df[df['classifier'] == clf]['test_accuracy'].values
        print(f"  {classifier_display_name(clf)}: mean={clf_acc.mean():.4f}, std={clf_acc.std():.4f}, n={len(clf_acc)}")
    
    print("="*80 + "\n")

def print_probe_metadata(probe_path):
    """Print detailed metadata for a specific probe."""
    metadata_path = Path(probe_path) / "probe_metadata.json"
    
    if not metadata_path.exists():
        print(f"⚠️  Metadata not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("="*80)
    print(f"PROBE METADATA: {metadata.get('feature_method', 'N/A')}_{metadata.get('classifier', 'N/A')}")
    print("="*80)
    print(json.dumps(metadata, indent=2))
    print("="*80 + "\n")

def create_visualizations(df):
    """Create visualization plots if matplotlib is available."""
    if not HAS_PLOTTING:
        return
    
    print("📊 Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Main 3x3 Heatmap - ALL 9 COMBINATIONS (MOST IMPORTANT)
    ax1 = fig.add_subplot(gs[0, :])
    pivot = df.pivot_table(
        values='test_accuracy',
        index='feature_method',
        columns='classifier',
        aggfunc='first'
    )
    # Reorder for consistency
    feature_order = ['SLT', 'TBG', 'LAST']
    classifier_order = ['random_forest', 'mlp', 'deep_nn']
    pivot = pivot.reindex(index=feature_order, columns=classifier_order)
    
    # Create heatmap with annotations
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax1, 
                cbar_kws={'label': 'Test Accuracy', 'shrink': 0.8}, vmin=0.5, vmax=1.0,
                linewidths=2, linecolor='black')
    ax1.set_title('ALL 9 COMBINATIONS: Test Accuracy (Feature Method × Classifier)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature Method', fontsize=12, fontweight='bold')
    ax1.set_xticklabels([classifier_display_name(x.get_text()) for x in ax1.get_xticklabels()], 
                        rotation=0, ha='center')
    ax1.set_yticklabels(['SLT (Second-to-Last)', 'TBG (Token Before Gen)', 'LAST (Last Token)'], 
                        rotation=0, ha='right')
    
    # 2. AUROC Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    pivot_auc = df.pivot_table(
        values='test_auc',
        index='feature_method',
        columns='classifier',
        aggfunc='first'
    )
    pivot_auc = pivot_auc.reindex(index=feature_order, columns=classifier_order)
    sns.heatmap(pivot_auc, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax2,
                cbar_kws={'label': 'Test AUROC', 'shrink': 0.8}, vmin=0.7, vmax=1.0,
                linewidths=2, linecolor='black')
    ax2.set_title('Test AUROC (Feature Method × Classifier)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Classifier', fontsize=10)
    ax2.set_ylabel('Feature Method', fontsize=10)
    ax2.set_xticklabels([classifier_display_name(x.get_text()) for x in ax2.get_xticklabels()], 
                       rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(['SLT', 'TBG', 'LAST'], rotation=0, ha='right', fontsize=9)
    
    # 3. Bar chart - All 9 combinations ranked
    ax3 = fig.add_subplot(gs[1, 1])
    df_sorted = df.sort_values('test_accuracy', ascending=True)
    colors = plt.cm.RdYlGn(df_sorted['test_accuracy'].values / df_sorted['test_accuracy'].max())
    y_pos = np.arange(len(df_sorted))
    labels = [f"{row['feature_method']} + {classifier_display_name(row['classifier'])}" 
              for _, row in df_sorted.iterrows()]
    bars = ax3.barh(y_pos, df_sorted['test_accuracy'].values, color=colors, edgecolor='black', linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Test Accuracy', fontsize=10, fontweight='bold')
    ax3.set_title('All 9 Combinations Ranked by Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlim(0.5, 1.0)
    ax3.grid(True, alpha=0.3, axis='x')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_sorted['test_accuracy'].values)):
        ax3.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 4. Box plot by feature method
    ax4 = fig.add_subplot(gs[2, 0])
    sns.boxplot(data=df, x='feature_method', y='test_accuracy', ax=ax4, 
                order=feature_order, palette='Set2')
    ax4.set_title('Test Accuracy Distribution by Feature Method', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=10)
    ax4.set_xlabel('Feature Method', fontsize=10)
    ax4.set_xticklabels(['SLT', 'TBG', 'LAST'], fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Box plot by classifier
    ax5 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=df, x='classifier', y='test_accuracy', ax=ax5,
                order=classifier_order, palette='Set3')
    ax5.set_title('Test Accuracy Distribution by Classifier', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_xlabel('Classifier', fontsize=10)
    ax5.set_xticklabels([classifier_display_name(x) for x in classifier_order], 
                       rotation=45, ha='right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Complete Analysis: All 9 Probe Combinations', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_file = RESULTS_DIR / "probe_analysis_plots.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualizations saved to: {output_file}")
    plt.close()

def print_recommendations(df):
    """Print recommendations based on results."""
    print("="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    best = df.loc[df['test_accuracy'].idxmax()]
    
    print(f"\n1. Best Overall Combination:")
    print(f"   Use {best['feature_method']} features with {classifier_display_name(best['classifier'])}")
    print(f"   Expected accuracy: {best['test_accuracy']:.1%}")
    
    # Check if results are reliable
    min_test_size = df['n_test'].min()
    if min_test_size < 5:
        print(f"\n⚠️  WARNING: Small test set (min={min_test_size} examples)")
        print(f"   Results may not be reliable. Consider running with more tasks.")
    
    # Check accuracy range
    acc_range = df['test_accuracy'].max() - df['test_accuracy'].min()
    if acc_range < 0.05:
        print(f"\n📊 All methods perform similarly (range: {acc_range:.3f})")
        print(f"   Consider other factors: inference speed, model size, etc.")
    else:
        print(f"\n📊 Significant variation in performance (range: {acc_range:.3f})")
        print(f"   Best method is {((df['test_accuracy'].max() - df['test_accuracy'].min()) / df['test_accuracy'].min() * 100):.1f}% better than worst")
    
    # Check if deep NN is worth it
    if 'deep_nn' in df['classifier'].values:
        deep_nn_acc = df[df['classifier'] == 'deep_nn']['test_accuracy'].max()
        other_acc = df[df['classifier'] != 'deep_nn']['test_accuracy'].max()
        if deep_nn_acc > other_acc + 0.02:
            print(f"\n✅ Deep NN provides meaningful improvement (+{deep_nn_acc - other_acc:.3f})")
        else:
            print(f"\n💭 Deep NN doesn't provide significant improvement")
            print(f"   Consider using simpler classifier for faster inference")
    
    print("="*80 + "\n")

# ============================================================
# Main Analysis
# ============================================================

def main():
    """Run complete analysis."""
    print("\n" + "="*80)
    print("PROBE EXPERIMENT RESULTS ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    df, results_json = load_results()
    if df is None:
        return
    
    # Run analyses
    print_summary(df)
    print_all_9_combinations(df)  # NEW: Clear 3x3 comparison table
    print_all_results(df)
    print_best_combination(df)
    print_comparison_by_feature_method(df)
    print_comparison_by_classifier(df)
    print_statistical_significance(df)
    print_recommendations(df)
    
    # Print best probe metadata if available
    best = df.loc[df['test_accuracy'].idxmax()]
    if 'probe_path' in best and Path(best['probe_path']).exists():
        print_probe_metadata(best['probe_path'])
    
    # Create visualizations
    if HAS_PLOTTING:
        create_visualizations(df)
    else:
        print("💡 Install matplotlib and seaborn for visualizations:")
        print("   pip install matplotlib seaborn")
    
    print("✅ Analysis complete!\n")

if __name__ == "__main__":
    main()

