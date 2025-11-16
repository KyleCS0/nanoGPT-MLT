#!/usr/bin/env python3
"""
Professional benchmark plotting script for nanoGPT.
Generates publication-quality plots for each benchmark type.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Set publication-quality style (serif, minimal, consistent)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.grid': True,
})


def load_results(log_file):
    """Load and parse benchmark results from JSONL file."""
    results = defaultdict(list)
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                benchmark_name = record['benchmark_name']
                results[benchmark_name].append(record)
    
    return results


def quadratic_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, deg=2)
    return coeffs  # a, b, c


def plot_latency_vs_T(data, output_dir):
    """
    Plot Latency vs T benchmark results.
    Creates separate standalone plots for total time and per-token time.
    """
    # Sort by T value
    data = sorted(data, key=lambda x: x['T'])
    
    T_values = [d['T'] for d in data]
    time_total_median = [d['time_total_ms_median'] for d in data]
    time_total_std = [d['time_total_ms_std'] for d in data]
    time_per_token = [d['time_per_token_ms_median'] for d in data]
    
    # Plot 1: Total Time vs T (standalone)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    ax1.errorbar(T_values, time_total_median, yerr=time_total_std, fmt='o',
                 markersize=4.5, elinewidth=1.0, capsize=2,
                 color='black', ecolor='#888888', label='Median ± std')
    # Quadratic fit to highlight scaling
    a, b, c = quadratic_fit(T_values, time_total_median)
    T_dense = np.linspace(min(T_values), max(T_values), 256)
    ax1.plot(T_dense, a*T_dense**2 + b*T_dense + c, '-', color='#2E86AB', linewidth=2,
             label='Quadratic fit')
    
    ax1.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax1.set_ylabel('Total Generation Time (ms)', fontweight='bold')
    ax1.set_title('Total Latency vs Generation Length', fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(frameon=False)
    ax1.set_xlim(left=min(T_values)*0.9, right=max(T_values)*1.05)
    
    # Add system info
    if data:
        info = data[0]
        fig1.text(0.5, 0.02, 
                f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | "
                f"dtype: {info['dtype']} | Batch Size: {info['batch_size']}",
                ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output_path1 = Path(output_dir) / 'latency_total_vs_T.png'
    plt.savefig(output_path1, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path1}")
    plt.close()
    
    # Plot 2: Time per Token vs T (standalone)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    ax2.plot(T_values, time_per_token, 's-', linewidth=1.8, markersize=4.5,
             color='#A23B72')
    
    ax2.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax2.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax2.set_title('Per-Token Latency vs Generation Length', fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(left=min(T_values)*0.9, right=max(T_values)*1.05)
    
    # Add system info
    if data:
        info = data[0]
        fig2.text(0.5, 0.02, 
                f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | "
                f"dtype: {info['dtype']} | Batch Size: {info['batch_size']}",
                ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output_path2 = Path(output_dir) / 'latency_per_token_vs_T.png'
    plt.savefig(output_path2, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path2}")
    plt.close()


def plot_vram_vs_T(data, output_dir):
    """
    Plot VRAM usage vs T: relative growth from first T point (in MB).
    """
    # Sort by T value
    data = sorted(data, key=lambda x: x['T'])
    
    T_values = [d['T'] for d in data]
    peak_memory_mb = [d['peak_memory_bytes'] / 1e6 for d in data]
    
    # Baseline subtract: use first T point as baseline
    baseline_memory_mb = peak_memory_mb[0]
    relative_memory_mb = [mem - baseline_memory_mb for mem in peak_memory_mb]
    
    # Create standalone figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(T_values, relative_memory_mb, 'o-', linewidth=1.8, markersize=4.5,
            color='#2E86AB')
    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Additional Memory Usage (MB)', fontweight='bold')
    ax.set_title('GPU Memory Growth vs Generation Length', fontweight='bold', pad=15)
    ax.set_xlim(left=min(T_values)*0.9, right=max(T_values)*1.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add system info
    if data:
        total_vram_mb = data[0]['gpu_total_vram'] / 1e6
        info = data[0]
        fig.text(0.5, 0.02,
                 f"GPU: {info['gpu_name']} ({total_vram_mb:.0f} MB) | "
                 f"dtype: {info['dtype']} | Batch Size: {info['batch_size']} | "
                 f"Baseline (T={T_values[0]}): {baseline_memory_mb:.1f} MB",
                 ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Save plot
    output_path = Path(output_dir) / 'vram_vs_T.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    
    plt.close()


def plot_per_phase_timing(data, output_dir):
    """
    Plot per-phase timing breakdown.
    Creates separate bar and pie chart figures.
    """
    if not data:
        print("  No per-phase timing data found.")
        return
    
    # Get the most recent result
    record = data[-1]
    phase_times = record['time_phase_ms']
    T_star = record['T_star']
    
    # Define phase order and colors
    phase_order = ['embedding', 'attention', 'mlp', 'head', 'other']
    phase_labels = {
        'embedding': 'Embedding',
        'attention': 'Self-Attention',
        'mlp': 'MLP',
        'head': 'LM Head',
        'other': 'Other'
    }
    phase_colors = {
        'embedding': '#06A77D',
        'attention': '#2E86AB',
        'mlp': '#A23B72',
        'head': '#F18F01',
        'other': '#999999'
    }
    
    # Extract data
    phases = [p for p in phase_order if p in phase_times and p != 'total']
    times = [phase_times[p] for p in phases]
    labels = [phase_labels[p] for p in phases]
    colors = [phase_colors[p] for p in phases]
    
    total_measured = sum(times)
    percentages = [100 * t / total_measured for t in times]
    
    # Bar chart (standalone)
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    bars = ax_bar.bar(labels, times, color=colors, edgecolor='white', linewidth=1.5)
    ax_bar.set_ylabel('Total Time (ms)', fontweight='bold')
    ax_bar.set_title(f'Phase Timing Breakdown (T={T_star})', fontweight='bold', pad=15)
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_bar.set_ylim(bottom=0, top=max(times) * 1.15)
    ax_bar.tick_params(axis='x', rotation=15)
    
    for bar, time, pct in zip(bars, times, percentages):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f} ms', ha='center', va='bottom', fontsize=9)
    
    # Add system info
    info = record
    fig_bar.text(0.5, 0.02,
                f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | "
                f"dtype: {info['dtype']} | Total Time: {total_measured:.1f} ms",
                ha='center', fontsize=9, style='italic', color='#555555')
    
    fig_bar.tight_layout(rect=[0, 0.04, 1, 1])
    bars_path = Path(output_dir) / 'per_phase_breakdown_bar.png'
    fig_bar.savefig(bars_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {bars_path}")
    plt.close(fig_bar)

    # Pie chart (standalone) - hide labels <1%
    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    explode = [0.05] * len(phases)
    
    # Custom autopct to hide small percentages
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct >= 1.0 else ''
    
    wedges, texts, autotexts = ax_pie.pie(percentages, labels=labels, colors=colors,
                                          autopct=autopct_format, startangle=90,
                                          explode=explode,
                                          wedgeprops=dict(edgecolor='white', linewidth=1.5))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax_pie.set_title(f'Phase Time Distribution (T={T_star})', fontweight='bold', pad=15)
    
    # Add system info
    fig_pie.text(0.5, 0.02,
                f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | "
                f"dtype: {info['dtype']} | Total Time: {total_measured:.1f} ms",
                ha='center', fontsize=9, style='italic', color='#555555')
    
    fig_pie.tight_layout(rect=[0, 0.04, 1, 1])
    pies_path = Path(output_dir) / 'per_phase_breakdown_pie.png'
    fig_pie.savefig(pies_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {pies_path}")
    plt.close(fig_pie)


def main():
    parser = argparse.ArgumentParser(
        description="Generate professional plots from nanoGPT benchmark results."
    )
    parser.add_argument(
        '--results',
        type=str,
        default='benchmark/results.jsonl',
        help='Path to the results JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark/plots',
        help='Directory to save the plots'
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    if not results:
        print("No benchmark results found!")
        return
    
    print(f"Found {len(results)} benchmark types")
    print()
    
    # Generate plots for each benchmark type
    if 'latency_vs_T' in results:
        print("Generating Latency vs T plots...")
        plot_latency_vs_T(results['latency_vs_T'], output_dir)
    
    if 'vram_vs_T' in results:
        print("Generating VRAM vs T plot...")
        plot_vram_vs_T(results['vram_vs_T'], output_dir)
    
    if 'per_phase_timing' in results:
        print("Generating Per-phase Timing plots...")
        plot_per_phase_timing(results['per_phase_timing'], output_dir)
    
    print()
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
