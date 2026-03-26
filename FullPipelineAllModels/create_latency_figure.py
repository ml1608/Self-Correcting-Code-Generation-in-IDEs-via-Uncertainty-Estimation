#!/usr/bin/env python3
"""
Create pass@k vs latency trade-off figure for Overleaf.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Data from results (SLT features)
# Format: (latency_multiplier, pass_at_1, model_name, method)
data = [
    # Baseline
    (1.00, 0.360, 'Llama-3.2-3B', 'Baseline'),
    (1.00, 0.760, 'Qwen2.5-Coder-3B', 'Baseline'),
    (1.00, 0.560, 'DeepSeek-Coder-1.3B', 'Baseline'),
    
    # SLT + temp sampling (Self-Correction)
    (2.78, 0.440, 'Llama-3.2-3B', 'SLT+temp sampling'),
    (2.86, 0.920, 'Qwen2.5-Coder-3B', 'SLT+temp sampling'),
    (2.22, 0.640, 'DeepSeek-Coder-1.3B', 'SLT+temp sampling'),
    
    # SLT + adaptive decoding
    (21.1, 0.480, 'Llama-3.2-3B', 'SLT+adaptive'),
    (21.6, 0.680, 'Qwen2.5-Coder-3B', 'SLT+adaptive'),
    (3.63, 0.480, 'DeepSeek-Coder-1.3B', 'SLT+adaptive'),
]

# Separate data by method
baseline_data = [(x, y, m) for x, y, m, method in data if method == 'Baseline']
temp_sampling_data = [(x, y, m) for x, y, m, method in data if method == 'SLT+temp sampling']
adaptive_data = [(x, y, m) for x, y, m, method in data if method == 'SLT+adaptive']

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot each method with different markers and colors
methods = [
    (baseline_data, 'Baseline (no regen)', 'o', '#1f77b4', 'black'),
    (temp_sampling_data, 'SLT + temp sampling', 's', '#2ca02c', 'black'),
    (adaptive_data, 'SLT + adaptive decoding', '^', '#d62728', 'black'),
]

for method_data, label, marker, color, edge_color in methods:
    x_vals = [x for x, y, m in method_data]
    y_vals = [y for x, y, m in method_data]
    models = [m for x, y, m in method_data]
    
    # Plot points
    ax.scatter(x_vals, y_vals, marker=marker, s=100, c=color, 
               edgecolors=edge_color, linewidths=1.5, label=label, zorder=3)
    
    # Connect points with lines (one curve per method)
    if len(x_vals) > 1:
        # Sort by latency for smooth curve
        sorted_data = sorted(zip(x_vals, y_vals), key=lambda t: t[0])
        sorted_x = [x for x, y in sorted_data]
        sorted_y = [y for x, y in sorted_data]
        ax.plot(sorted_x, sorted_y, color=color, linestyle='--', 
                alpha=0.5, linewidth=1.5, zorder=2)
    
    # Add model labels
    for x, y, model in method_data:
        # Shorten model names for readability
        short_name = model.replace('-3B', '').replace('-Coder-1.3B', '')
        ax.annotate(short_name, (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, alpha=0.7)

# Set labels and title
ax.set_xlabel('Latency Multiplier (× baseline)', fontweight='bold')
ax.set_ylabel('Pass@1', fontweight='bold')
ax.set_title('Pass@1 vs Latency Trade-off', fontweight='bold', pad=15)

# Set x-axis to log scale for better visualization (adaptive decoding has high latency)
ax.set_xscale('log')
ax.set_xlim(0.8, 30)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

# Improve layout
plt.tight_layout()

# Save as PDF (best for LaTeX) and PNG (for preview)
script_dir = Path(__file__).parent
output_dir = script_dir / 'figures'
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / 'passk_latency_tradeoff.pdf', format='pdf', bbox_inches='tight')
plt.savefig(output_dir / 'passk_latency_tradeoff.png', format='png', bbox_inches='tight')

print(f"✅ Figure saved to {output_dir / 'passk_latency_tradeoff.pdf'}")
print(f"✅ Figure saved to {output_dir / 'passk_latency_tradeoff.png'}")
print(f"\nFigure location: {output_dir.absolute()}")

