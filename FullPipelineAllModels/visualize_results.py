#!/usr/bin/env python3
"""
Visualize pipeline results: latency and improvements for TBG and SLT methods.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_summary_data(results):
    """Extract summary data from results."""
    summary = []
    
    for result in results:
        model_id = result["model_id"]
        feature_method = result["feature_method"]
        
        # Baseline
        baseline = result.get("baseline_results", {})
        baseline_pass = baseline.get("pass_at_1", 0)
        baseline_latency = baseline.get("avg_latency", 0)
        
        # Adaptive Decoding
        adaptive = result.get("adaptive_results", {})
        adaptive_pass = adaptive.get("adaptive_pass_at_1", 0)
        adaptive_improvement = adaptive.get("improvement", 0)
        adaptive_latency = adaptive.get("avg_adaptive_latency", 0)
        adaptive_ratio = adaptive.get("avg_adaptive_ratio", 0)
        
        # Self-Correction
        correction = result.get("correction_results", {})
        correction_pass = correction.get("corrected_pass_at_1", 0)
        correction_improvement = correction.get("improvement", 0)
        correction_latency = correction.get("avg_corrected_latency", 0)
        correction_attempts = correction.get("avg_num_corrections", 0)
        
        summary.append({
            "Model": model_id.split("/")[-1].replace("-Instruct", ""),
            "Feature Method": feature_method,
            "Baseline Pass@1": baseline_pass,
            "Baseline Latency (s)": baseline_latency,
            "Adaptive Pass@1": adaptive_pass,
            "Adaptive Improvement": adaptive_improvement,
            "Adaptive Latency (s)": adaptive_latency,
            "Adaptive Ratio": adaptive_ratio,
            "Correction Pass@1": correction_pass,
            "Correction Improvement": correction_improvement,
            "Correction Latency (s)": correction_latency,
            "Avg Corrections": correction_attempts,
        })
    
    return pd.DataFrame(summary)

def create_summary_table(df, output_path):
    """Create a formatted summary table."""
    # Create a more readable table
    table_data = []
    
    for _, row in df.iterrows():
        model = row["Model"]
        method = row["Feature Method"]
        
        table_data.append({
            "Model": model,
            "Method": method,
            "Baseline": f"{row['Baseline Pass@1']:.3f} ({row['Baseline Latency (s)']:.2f}s)",
            "Adaptive": f"{row['Adaptive Pass@1']:.3f} ({row['Adaptive Improvement']:+.3f}, {row['Adaptive Latency (s)']:.2f}s)",
            "Self-Correct": f"{row['Correction Pass@1']:.3f} ({row['Correction Improvement']:+.3f}, {row['Correction Latency (s)']:.2f}s)",
        })
    
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_df.to_csv(output_path / "results_summary_table.csv", index=False)
    
    # Print formatted table
    print("\n" + "="*120)
    print("RESULTS SUMMARY TABLE")
    print("="*120)
    print(table_df.to_string(index=False))
    print("="*120 + "\n")
    
    return table_df

def create_visualizations(df, output_path):
    """Create visualizations."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Pass@1 Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :])
    models = df["Model"].unique()
    x = np.arange(len(models))
    width = 0.35
    
    for method in ["SLT", "TBG"]:
        method_df = df[df["Feature Method"] == method]
        baseline = [method_df[method_df["Model"] == m]["Baseline Pass@1"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        adaptive = [method_df[method_df["Model"] == m]["Adaptive Pass@1"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        correction = [method_df[method_df["Model"] == m]["Correction Pass@1"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        
        offset = -width if method == "SLT" else width
        ax1.bar(x + offset - width/2, baseline, width/3, label=f"{method} Baseline", alpha=0.7)
        ax1.bar(x + offset, adaptive, width/3, label=f"{method} Adaptive", alpha=0.7)
        ax1.bar(x + offset + width/2, correction, width/3, label=f"{method} Self-Correct", alpha=0.7)
    
    ax1.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Pass@1", fontsize=12, fontweight='bold')
    ax1.set_title("Pass@1 Comparison: Baseline vs Adaptive vs Self-Correction", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(ncol=3, loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement Comparison (Bar Chart)
    ax2 = fig.add_subplot(gs[1, 0])
    methods = ["SLT", "TBG"]
    models = df["Model"].unique()
    
    adaptive_improvements = []
    correction_improvements = []
    
    for method in methods:
        method_df = df[df["Feature Method"] == method]
        adaptive_imp = [method_df[method_df["Model"] == m]["Adaptive Improvement"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        correction_imp = [method_df[method_df["Model"] == m]["Correction Improvement"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        adaptive_improvements.append(adaptive_imp)
        correction_improvements.append(correction_imp)
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, method in enumerate(methods):
        ax2.bar(x + i*width - width/2, adaptive_improvements[i], width, label=f"{method} Adaptive", alpha=0.7)
        ax2.bar(x + i*width, correction_improvements[i], width, label=f"{method} Self-Correct", alpha=0.7)
    
    ax2.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Improvement (Pass@1)", fontsize=12, fontweight='bold')
    ax2.set_title("Improvement Over Baseline", fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(ncol=2, fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Latency Comparison (Bar Chart)
    ax3 = fig.add_subplot(gs[1, 1])
    
    baseline_latencies = []
    adaptive_latencies = []
    correction_latencies = []
    
    for method in methods:
        method_df = df[df["Feature Method"] == method]
        baseline_lat = [method_df[method_df["Model"] == m]["Baseline Latency (s)"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        adaptive_lat = [method_df[method_df["Model"] == m]["Adaptive Latency (s)"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        correction_lat = [method_df[method_df["Model"] == m]["Correction Latency (s)"].values[0] if len(method_df[method_df["Model"] == m]) > 0 else 0 for m in models]
        baseline_latencies.append(baseline_lat)
        adaptive_latencies.append(adaptive_lat)
        correction_latencies.append(correction_lat)
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, method in enumerate(methods):
        offset = i * width - width
        ax3.bar(x + offset, baseline_latencies[i], width, label=f"{method} Baseline", alpha=0.7)
        ax3.bar(x + offset + width, adaptive_latencies[i], width, label=f"{method} Adaptive", alpha=0.7)
        ax3.bar(x + offset + 2*width, correction_latencies[i], width, label=f"{method} Self-Correct", alpha=0.7)
    
    ax3.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Latency (seconds)", fontsize=12, fontweight='bold')
    ax3.set_title("Latency Comparison", fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(ncol=3, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement Heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Create heatmap data
    heatmap_data = []
    for model in models:
        row = []
        for method in methods:
            method_df = df[(df["Feature Method"] == method) & (df["Model"] == model)]
            if len(method_df) > 0:
                row.append(method_df["Correction Improvement"].values[0])
            else:
                row.append(0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=models, columns=methods)
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                ax=ax4, cbar_kws={'label': 'Improvement'})
    ax4.set_title("Self-Correction Improvement Heatmap", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Feature Method", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Model", fontsize=12, fontweight='bold')
    
    # 5. Latency vs Improvement Scatter
    ax5 = fig.add_subplot(gs[2, 1])
    
    for method in methods:
        method_df = df[df["Feature Method"] == method]
        ax5.scatter(method_df["Correction Latency (s)"], method_df["Correction Improvement"],
                   s=100, alpha=0.6, label=f"{method} Self-Correct")
        ax5.scatter(method_df["Adaptive Latency (s)"], method_df["Adaptive Improvement"],
                   s=100, alpha=0.6, marker='^', label=f"{method} Adaptive")
    
    ax5.set_xlabel("Latency (seconds)", fontsize=12, fontweight='bold')
    ax5.set_ylabel("Improvement (Pass@1)", fontsize=12, fontweight='bold')
    ax5.set_title("Latency vs Improvement Trade-off", fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax5.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.suptitle("Full Pipeline Results: All Models, TBG vs SLT", fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path / "pipeline_results_visualization.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path / 'pipeline_results_visualization.png'}")
    
    return fig

def main():
    """Main function."""
    # Find the results file
    results_dir = Path(__file__).parent / "pipeline_results_all_models"
    json_files = list(results_dir.glob("all_models_results_*.json"))
    
    if not json_files:
        print("❌ No results JSON file found!")
        return
    
    # Use the most recent one
    json_path = sorted(json_files)[-1]
    print(f"📊 Loading results from: {json_path.name}")
    
    # Load and process
    results = load_results(json_path)
    df = extract_summary_data(results)
    
    print(f"\n✅ Loaded {len(df)} result entries")
    print(f"   Models: {df['Model'].unique()}")
    print(f"   Methods: {df['Feature Method'].unique()}")
    
    # Create output directory
    output_dir = results_dir
    output_dir.mkdir(exist_ok=True)
    
    # Create summary table
    table_df = create_summary_table(df, output_dir)
    
    # Create visualizations
    fig = create_visualizations(df, output_dir)
    
    # Print summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    for method in ["SLT", "TBG"]:
        method_df = df[df["Feature Method"] == method]
        print(f"\n{method} Method:")
        print(f"  Baseline Pass@1: {method_df['Baseline Pass@1'].mean():.3f} ± {method_df['Baseline Pass@1'].std():.3f}")
        print(f"  Adaptive Improvement: {method_df['Adaptive Improvement'].mean():+.3f} ± {method_df['Adaptive Improvement'].std():.3f}")
        print(f"  Self-Correction Improvement: {method_df['Correction Improvement'].mean():+.3f} ± {method_df['Correction Improvement'].std():.3f}")
        print(f"  Baseline Latency: {method_df['Baseline Latency (s)'].mean():.2f}s ± {method_df['Baseline Latency (s)'].std():.2f}s")
        print(f"  Adaptive Latency: {method_df['Adaptive Latency (s)'].mean():.2f}s ± {method_df['Adaptive Latency (s)'].std():.2f}s")
        print(f"  Correction Latency: {method_df['Correction Latency (s)'].mean():.2f}s ± {method_df['Correction Latency (s)'].std():.2f}s")
    
    print("\n" + "="*120)
    print("✅ Analysis complete!")
    print(f"   - Summary table: {output_dir / 'results_summary_table.csv'}")
    print(f"   - Visualization: {output_dir / 'pipeline_results_visualization.png'}")
    print("="*120 + "\n")

if __name__ == "__main__":
    main()


