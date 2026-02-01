#!/usr/bin/env python3
"""
Comprehensive Threshold Analysis and Visualization

This script analyzes threshold tuning results and provides:
1. Visualizations comparing all thresholds
2. Best threshold recommendations for each probe
3. Summary analysis across all probes
4. Detailed comparison tables
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent
THRESHOLD_ANALYSIS_DIR = SCRIPT_DIR / "threshold_analysis"
OUTPUT_DIR = SCRIPT_DIR / "threshold_analysis_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Thresholds that were tested
THRESHOLDS_TO_TEST = [0.3, 0.4, 0.5, 0.6, 0.7]

# ============================================================
# Load Results
# ============================================================

def load_all_threshold_results(analysis_dir: Path):
    """Load all threshold analysis JSON files."""
    results = []
    
    if not analysis_dir.exists():
        print(f"❌ Threshold analysis directory not found: {analysis_dir}")
        return results
    
    json_files = list(analysis_dir.glob("*_threshold_analysis.json"))
    
    if len(json_files) == 0:
        print(f"⚠️  No threshold analysis JSON files found in {analysis_dir}")
        return results
    
    print(f"Found {len(json_files)} threshold analysis file(s)\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            results.append(data)
            print(f"✅ Loaded: {json_file.name}")
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    return results

# ============================================================
# Analysis Functions
# ============================================================

def find_best_threshold(result_data):
    """Find the best threshold based on F1 score."""
    threshold_results = result_data.get("threshold_results", [])
    if not threshold_results:
        return None
    
    best_idx = np.argmax([r['f1'] for r in threshold_results])
    return threshold_results[best_idx]

def analyze_threshold_tradeoffs(result_data):
    """Analyze tradeoffs between precision, recall, and trigger rate."""
    threshold_results = result_data.get("threshold_results", [])
    if not threshold_results:
        return None
    
    analysis = {
        "thresholds": [r['threshold'] for r in threshold_results],
        "f1_scores": [r['f1'] for r in threshold_results],
        "precision_scores": [r['precision'] for r in threshold_results],
        "recall_scores": [r['recall'] for r in threshold_results],
        "accuracy_scores": [r['accuracy'] for r in threshold_results],
        "trigger_rates": [r['trigger_rate'] for r in threshold_results],
    }
    
    return analysis

# ============================================================
# Visualization Functions
# ============================================================

def plot_threshold_comparison(result_data, output_dir: Path):
    """Create comprehensive visualization for a single probe."""
    probe_name = result_data["probe_name"]
    threshold_results = result_data.get("threshold_results", [])
    recommended_threshold = result_data.get("current_recommended_threshold", 0.5)
    best_threshold_data = result_data.get("best_threshold", {})
    
    if not threshold_results:
        print(f"⚠️  No threshold results for {probe_name}, skipping plot")
        return
    
    # Extract data
    thresholds = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1_scores = [r['f1'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    trigger_rates = [r['trigger_rate'] * 100 for r in threshold_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Threshold Analysis: {probe_name}", fontsize=16, fontweight='bold')
    
    # 1. Precision, Recall, F1, Accuracy vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2, markersize=8)
    ax1.plot(thresholds, recalls, 's-', label='Recall', linewidth=2, markersize=8)
    ax1.plot(thresholds, f1_scores, '^-', label='F1', linewidth=2, markersize=8)
    ax1.plot(thresholds, accuracies, 'd-', label='Accuracy', linewidth=2, markersize=8)
    ax1.axvline(recommended_threshold, color='red', linestyle='--', 
                label=f'Recommended ({recommended_threshold:.3f})', linewidth=2)
    if best_threshold_data:
        ax1.axvline(best_threshold_data.get('threshold', 0.5), color='green', 
                   linestyle='--', label=f'Best F1 ({best_threshold_data.get("threshold", 0.5):.1f})', 
                   linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Threshold', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thresholds)
    
    # 2. Trigger Rate vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(thresholds, trigger_rates, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axvline(recommended_threshold, color='red', linestyle='--', 
                label=f'Recommended ({recommended_threshold:.3f})', linewidth=2)
    if best_threshold_data:
        ax2.axvline(best_threshold_data.get('threshold', 0.5), color='green', 
                   linestyle='--', label=f'Best F1 ({best_threshold_data.get("threshold", 0.5):.1f})', 
                   linewidth=2)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Trigger Rate (%)', fontsize=12)
    ax2.set_title('Correction Trigger Rate vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(thresholds)
    
    # 3. Precision-Recall Tradeoff
    ax3 = axes[0, 2]
    ax3.plot(recalls, precisions, 'o-', linewidth=2, markersize=8, color='teal')
    for i, thresh in enumerate(thresholds):
        ax3.annotate(f'{thresh:.1f}', (recalls[i], precisions[i]), 
                    fontsize=9, ha='center', va='bottom')
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Tradeoff', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix Heatmap for Best Threshold
    ax4 = axes[1, 0]
    best_idx = np.argmax(f1_scores)
    best_result = threshold_results[best_idx]
    cm_data = [
        [best_result['true_negatives'], best_result['false_positives']],
        [best_result['false_negatives'], best_result['true_positives']]
    ]
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Low Entropy', 'High Entropy'],
                yticklabels=['Low Entropy', 'High Entropy'],
                cbar_kws={'label': 'Count'})
    ax4.set_title(f'Confusion Matrix (Best: {thresholds[best_idx]:.1f})', 
                  fontsize=13, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=11)
    ax4.set_xlabel('Predicted Label', fontsize=11)
    
    # 5. F1 Score Bar Chart
    ax5 = axes[1, 1]
    colors = ['green' if i == best_idx else 'steelblue' for i in range(len(thresholds))]
    bars = ax5.bar(range(len(thresholds)), f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_xticks(range(len(thresholds)))
    ax5.set_xticklabels([f'{t:.1f}' for t in thresholds])
    ax5.set_ylabel('F1 Score', fontsize=12)
    ax5.set_xlabel('Threshold', fontsize=12)
    ax5.set_title('F1 Score by Threshold', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    Model: {result_data.get('model_id', 'N/A')}
    Feature Method: {result_data.get('feature_method', 'N/A')}
    
    Current Recommended: {recommended_threshold:.4f}
    
    Best Threshold (F1): {best_threshold_data.get('threshold', 'N/A')}
    - F1:        {best_threshold_data.get('f1', 0):.4f}
    - Precision: {best_threshold_data.get('precision', 0):.4f}
    - Recall:    {best_threshold_data.get('recall', 0):.4f}
    - Accuracy:  {best_threshold_data.get('accuracy', 0):.4f}
    - Trigger:   {best_threshold_data.get('trigger_rate', 0)*100:.1f}%
    
    Test Set Size: {result_data.get('test_set_size', 'N/A')}
    
    Prediction Range:
    - Min:    {result_data.get('prediction_stats', {}).get('min', 0):.4f}
    - Median: {result_data.get('prediction_stats', {}).get('median', 0):.4f}
    - Max:    {result_data.get('prediction_stats', {}).get('max', 0):.4f}
    """
    ax6.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    
    output_file = output_dir / f"{probe_name}_threshold_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_file.name}")
    plt.close()

def plot_comparison_across_probes(all_results, output_dir: Path):
    """Create comparison plots across all probes."""
    if not all_results:
        return
    
    # Prepare data for comparison
    comparison_data = []
    for result in all_results:
        probe_name = result["probe_name"]
        model = result.get("model_id", "unknown")
        feature_method = result.get("feature_method", "unknown")
        
        for thresh_result in result.get("threshold_results", []):
            comparison_data.append({
                "probe": probe_name,
                "model": model.split("/")[-1] if "/" in model else model,
                "feature_method": feature_method,
                "threshold": thresh_result["threshold"],
                "f1": thresh_result["f1"],
                "precision": thresh_result["precision"],
                "recall": thresh_result["recall"],
                "accuracy": thresh_result["accuracy"],
                "trigger_rate": thresh_result["trigger_rate"] * 100,
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Threshold Comparison Across All Probes', fontsize=16, fontweight='bold')
    
    # 1. F1 Scores Heatmap
    ax1 = axes[0, 0]
    pivot_f1 = df.pivot_table(
        values='f1',
        index=['model', 'feature_method'],
        columns='threshold',
        aggfunc='first'
    )
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1,
                cbar_kws={'label': 'F1 Score'}, vmin=0.5, vmax=1.0,
                linewidths=1, linecolor='black')
    ax1.set_title('F1 Scores by Model/Method and Threshold', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Threshold', fontsize=11)
    ax1.set_ylabel('Model + Feature Method', fontsize=11)
    
    # 2. Trigger Rates Heatmap
    ax2 = axes[0, 1]
    pivot_trigger = df.pivot_table(
        values='trigger_rate',
        index=['model', 'feature_method'],
        columns='threshold',
        aggfunc='first'
    )
    sns.heatmap(pivot_trigger, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                cbar_kws={'label': 'Trigger Rate (%)'}, vmin=0, vmax=100,
                linewidths=1, linecolor='black')
    ax2.set_title('Trigger Rates (%) by Model/Method and Threshold', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('Model + Feature Method', fontsize=11)
    
    # 3. Best F1 per Probe
    ax3 = axes[1, 0]
    best_per_probe = df.loc[df.groupby(['model', 'feature_method'])['f1'].idxmax()]
    best_per_probe = best_per_probe.sort_values('f1', ascending=True)
    y_pos = np.arange(len(best_per_probe))
    labels = [f"{row['model']}\n{row['feature_method']}" for _, row in best_per_probe.iterrows()]
    colors = plt.cm.viridis(best_per_probe['f1'].values / best_per_probe['f1'].max())
    bars = ax3.barh(y_pos, best_per_probe['f1'].values, color=colors, edgecolor='black', linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Best F1 Score', fontsize=11)
    ax3.set_title('Best F1 Score per Probe', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, best_per_probe['f1'].values)):
        ax3.text(val + 0.01, i, f'{val:.3f} (t={best_per_probe.iloc[i]["threshold"]:.1f})',
                va='center', fontsize=8, fontweight='bold')
    
    # 4. Threshold Distribution (which threshold is best most often)
    ax4 = axes[1, 1]
    best_thresholds = df.loc[df.groupby(['model', 'feature_method'])['f1'].idxmax()]['threshold']
    threshold_counts = best_thresholds.value_counts().sort_index()
    bars = ax4.bar(threshold_counts.index, threshold_counts.values, 
                  color='steelblue', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Threshold', fontsize=11)
    ax4.set_ylabel('Number of Probes', fontsize=11)
    ax4.set_title('Best Threshold Distribution', fontsize=13, fontweight='bold')
    ax4.set_xticks(threshold_counts.index)
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, threshold_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / "threshold_comparison_across_probes.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {output_file.name}")
    plt.close()

# ============================================================
# Recommendation Functions
# ============================================================

def generate_threshold_recommendations(all_results):
    """Generate recommendations for best thresholds."""
    recommendations = []
    
    for result in all_results:
        probe_name = result["probe_name"]
        model = result.get("model_id", "unknown")
        feature_method = result.get("feature_method", "unknown")
        threshold_results = result.get("threshold_results", [])
        recommended_threshold = result.get("current_recommended_threshold", 0.5)
        best_threshold_data = result.get("best_threshold", {})
        
        if not threshold_results:
            continue
        
        # Find best by different metrics
        best_f1_idx = np.argmax([r['f1'] for r in threshold_results])
        best_prec_idx = np.argmax([r['precision'] for r in threshold_results])
        best_recall_idx = np.argmax([r['recall'] for r in threshold_results])
        
        # Find balanced threshold (closest to equal precision/recall)
        balanced_idx = None
        min_diff = float('inf')
        for i, r in enumerate(threshold_results):
            diff = abs(r['precision'] - r['recall'])
            if diff < min_diff:
                min_diff = diff
                balanced_idx = i
        
        recommendations.append({
            "probe": probe_name,
            "model": model,
            "feature_method": feature_method,
            "current_recommended": recommended_threshold,
            "best_f1": {
                "threshold": threshold_results[best_f1_idx]['threshold'],
                "f1": threshold_results[best_f1_idx]['f1'],
                "precision": threshold_results[best_f1_idx]['precision'],
                "recall": threshold_results[best_f1_idx]['recall'],
                "trigger_rate": threshold_results[best_f1_idx]['trigger_rate'],
            },
            "best_precision": {
                "threshold": threshold_results[best_prec_idx]['threshold'],
                "precision": threshold_results[best_prec_idx]['precision'],
                "recall": threshold_results[best_prec_idx]['recall'],
                "f1": threshold_results[best_prec_idx]['f1'],
            },
            "best_recall": {
                "threshold": threshold_results[best_recall_idx]['threshold'],
                "precision": threshold_results[best_recall_idx]['precision'],
                "recall": threshold_results[best_recall_idx]['recall'],
                "f1": threshold_results[best_recall_idx]['f1'],
            },
            "balanced": {
                "threshold": threshold_results[balanced_idx]['threshold'] if balanced_idx is not None else None,
                "precision": threshold_results[balanced_idx]['precision'] if balanced_idx is not None else None,
                "recall": threshold_results[balanced_idx]['recall'] if balanced_idx is not None else None,
                "f1": threshold_results[balanced_idx]['f1'] if balanced_idx is not None else None,
            } if balanced_idx is not None else None,
        })
    
    return recommendations

def print_recommendations(recommendations):
    """Print threshold recommendations."""
    print("\n" + "="*80)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*80 + "\n")
    
    for rec in recommendations:
        print(f"\n{'─'*80}")
        print(f"Probe: {rec['probe']}")
        print(f"Model: {rec['model']}")
        print(f"Feature Method: {rec['feature_method']}")
        print(f"{'─'*80}")
        
        print(f"\nCurrent Recommended Threshold: {rec['current_recommended']:.4f}")
        
        print(f"\n📊 BEST BY F1 SCORE (Recommended for overall performance):")
        best_f1 = rec['best_f1']
        print(f"   Threshold: {best_f1['threshold']:.1f}")
        print(f"   F1:        {best_f1['f1']:.4f}")
        print(f"   Precision: {best_f1['precision']:.4f}")
        print(f"   Recall:    {best_f1['recall']:.4f}")
        print(f"   Trigger Rate: {best_f1['trigger_rate']*100:.1f}%")
        
        print(f"\n🎯 BEST BY PRECISION (Fewer false positives):")
        best_prec = rec['best_precision']
        print(f"   Threshold: {best_prec['threshold']:.1f}")
        print(f"   Precision: {best_prec['precision']:.4f}")
        print(f"   Recall:    {best_prec['recall']:.4f}")
        print(f"   F1:        {best_prec['f1']:.4f}")
        
        print(f"\n🔍 BEST BY RECALL (Fewer false negatives):")
        best_recall = rec['best_recall']
        print(f"   Threshold: {best_recall['threshold']:.1f}")
        print(f"   Precision: {best_recall['precision']:.4f}")
        print(f"   Recall:    {best_recall['recall']:.4f}")
        print(f"   F1:        {best_recall['f1']:.4f}")
        
        if rec['balanced']:
            print(f"\n⚖️  BALANCED (Precision ≈ Recall):")
            balanced = rec['balanced']
            print(f"   Threshold: {balanced['threshold']:.1f}")
            print(f"   Precision: {balanced['precision']:.4f}")
            print(f"   Recall:    {balanced['recall']:.4f}")
            print(f"   F1:        {balanced['f1']:.4f}")
        
        print(f"\n💡 RECOMMENDATION:")
        if abs(best_f1['threshold'] - rec['current_recommended']) < 0.1:
            print(f"   Current recommended threshold ({rec['current_recommended']:.4f}) is close to best F1 ({best_f1['threshold']:.1f})")
            print(f"   ✅ Keep current threshold or use {best_f1['threshold']:.1f}")
        else:
            print(f"   Consider changing from {rec['current_recommended']:.4f} to {best_f1['threshold']:.1f}")
            print(f"   Expected improvement: F1 {best_f1['f1']:.4f} vs current")

# ============================================================
# Summary Tables
# ============================================================

def create_summary_tables(all_results, output_dir: Path):
    """Create summary tables in CSV format."""
    # Detailed table
    detailed_data = []
    for result in all_results:
        for thresh_result in result.get("threshold_results", []):
            detailed_data.append({
                "probe": result["probe_name"],
                "model": result.get("model_id", "unknown"),
                "feature_method": result.get("feature_method", "unknown"),
                "threshold": thresh_result["threshold"],
                "precision": thresh_result["precision"],
                "recall": thresh_result["recall"],
                "f1": thresh_result["f1"],
                "accuracy": thresh_result["accuracy"],
                "trigger_rate_pct": thresh_result["trigger_rate"] * 100,
                "true_positives": thresh_result["true_positives"],
                "false_positives": thresh_result["false_positives"],
                "true_negatives": thresh_result["true_negatives"],
                "false_negatives": thresh_result["false_negatives"],
            })
    
    df_detailed = pd.DataFrame(detailed_data)
    detailed_file = output_dir / "threshold_analysis_detailed.csv"
    df_detailed.to_csv(detailed_file, index=False)
    print(f"✅ Saved detailed table: {detailed_file.name}")
    
    # Best threshold per probe
    best_data = []
    for result in all_results:
        best_threshold_data = result.get("best_threshold", {})
        if best_threshold_data:
            best_data.append({
                "probe": result["probe_name"],
                "model": result.get("model_id", "unknown"),
                "feature_method": result.get("feature_method", "unknown"),
                "best_threshold": best_threshold_data.get("threshold", None),
                "best_f1": best_threshold_data.get("f1", None),
                "best_precision": best_threshold_data.get("precision", None),
                "best_recall": best_threshold_data.get("recall", None),
                "best_accuracy": best_threshold_data.get("accuracy", None),
                "best_trigger_rate_pct": best_threshold_data.get("trigger_rate", 0) * 100,
                "current_recommended": result.get("current_recommended_threshold", None),
            })
    
    df_best = pd.DataFrame(best_data)
    best_file = output_dir / "threshold_recommendations.csv"
    df_best.to_csv(best_file, index=False)
    print(f"✅ Saved recommendations table: {best_file.name}")
    
    return df_detailed, df_best

# ============================================================
# Main Analysis
# ============================================================

def main():
    """Run comprehensive threshold analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE THRESHOLD ANALYSIS")
    print("="*80 + "\n")
    
    # Load all results
    all_results = load_all_threshold_results(THRESHOLD_ANALYSIS_DIR)
    
    if len(all_results) == 0:
        print("❌ No threshold analysis results found.")
        print(f"   Expected files in: {THRESHOLD_ANALYSIS_DIR}")
        print("   Make sure you've run thresh_tune.py first.")
        return
    
    print(f"\n✅ Loaded {len(all_results)} probe analysis result(s)\n")
    
    # Generate individual plots
    print("="*80)
    print("GENERATING INDIVIDUAL PROBE VISUALIZATIONS")
    print("="*80 + "\n")
    
    for result in all_results:
        plot_threshold_comparison(result, OUTPUT_DIR)
    
    # Generate comparison plots
    print(f"\n{'='*80}")
    print("GENERATING CROSS-PROBE COMPARISON")
    print(f"{'='*80}\n")
    plot_comparison_across_probes(all_results, OUTPUT_DIR)
    
    # Generate recommendations
    print(f"\n{'='*80}")
    print("GENERATING RECOMMENDATIONS")
    print(f"{'='*80}\n")
    recommendations = generate_threshold_recommendations(all_results)
    print_recommendations(recommendations)
    
    # Create summary tables
    print(f"\n{'='*80}")
    print("CREATING SUMMARY TABLES")
    print(f"{'='*80}\n")
    df_detailed, df_best = create_summary_tables(all_results, OUTPUT_DIR)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    print("Best Threshold Distribution:")
    if 'best_threshold' in df_best.columns:
        threshold_dist = df_best['best_threshold'].value_counts().sort_index()
        for thresh, count in threshold_dist.items():
            print(f"  {thresh:.1f}: {count} probe(s)")
    
    print(f"\nAverage Best F1 Score: {df_best['best_f1'].mean():.4f}")
    print(f"Average Best Precision: {df_best['best_precision'].mean():.4f}")
    print(f"Average Best Recall: {df_best['best_recall'].mean():.4f}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"📊 Visualizations saved to: {OUTPUT_DIR}/")
    print(f"📋 Summary tables saved to: {OUTPUT_DIR}/")
    print(f"\n💡 Check 'threshold_recommendations.csv' for best threshold per probe")

if __name__ == "__main__":
    main()

