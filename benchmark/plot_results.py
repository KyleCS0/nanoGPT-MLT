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

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5


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


def plot_latency_vs_T(data, output_dir):
    """
    Plot Latency vs T benchmark results.
    Shows both total time and time per token.
    """
    # Sort by T value
    data = sorted(data, key=lambda x: x['T'])
    
    T_values = [d['T'] for d in data]
    time_total_median = [d['time_total_ms_median'] for d in data]
    time_total_mean = [d['time_total_ms_mean'] for d in data]
    time_total_std = [d['time_total_ms_std'] for d in data]
    time_per_token = [d['time_per_token_ms_median'] for d in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Total Time vs T
    ax1.plot(T_values, time_total_median, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label='Median', markeredgewidth=1.5, markeredgecolor='white')
    ax1.fill_between(T_values, 
                      np.array(time_total_mean) - np.array(time_total_std),
                      np.array(time_total_mean) + np.array(time_total_std),
                      alpha=0.2, color='#2E86AB', label='±1 std (mean)')
    
    ax1.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax1.set_ylabel('Total Generation Time (ms)', fontweight='bold')
    ax1.set_title('Total Latency vs Generation Length', fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_xlim(left=min(T_values)-10, right=max(T_values)+10)
    
    # Add value annotations
    for i, (t, val) in enumerate(zip(T_values, time_total_median)):
        ax1.annotate(f'{val:.1f}', 
                    xy=(t, val), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.7))
    
    # Plot 2: Time per Token vs T
    ax2.plot(T_values, time_per_token, 's-', linewidth=2, markersize=8,
             color='#A23B72', markeredgewidth=1.5, markeredgecolor='white')
    
    ax2.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax2.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax2.set_title('Per-Token Latency vs Generation Length', fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(left=min(T_values)-10, right=max(T_values)+10)
    
    # Add value annotations
    for i, (t, val) in enumerate(zip(T_values, time_per_token)):
        ax2.annotate(f'{val:.3f}', 
                    xy=(t, val), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.7))
    
    # Add system info as subtitle
    if data:
        info = data[0]
        fig.suptitle(f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | "
                    f"dtype: {info['dtype']} | Batch Size: {info['batch_size']}",
                    fontsize=10, y=0.98, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    output_path = Path(output_dir) / 'latency_vs_T.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    
    plt.close()


def plot_vram_vs_T(data, output_dir):
    """
    Plot VRAM usage vs T benchmark results.
    """
    # Sort by T value
    data = sorted(data, key=lambda x: x['T'])
    
    T_values = [d['T'] for d in data]
    peak_memory_gb = [d['peak_memory_bytes'] / 1e9 for d in data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with filled area
    ax.plot(T_values, peak_memory_gb, 'o-', linewidth=2.5, markersize=10,
            color='#F18F01', markeredgewidth=2, markeredgecolor='white')
    ax.fill_between(T_values, 0, peak_memory_gb, alpha=0.3, color='#F18F01')
    
    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Peak GPU Memory Usage (GB)', fontweight='bold')
    ax.set_title('GPU Memory Consumption vs Generation Length', fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(left=min(T_values)-10, right=max(T_values)+10)
    ax.set_ylim(bottom=0)
    
    # Add value annotations
    for i, (t, val) in enumerate(zip(T_values, peak_memory_gb)):
        ax.annotate(f'{val:.3f} GB', 
                   xy=(t, val), 
                   xytext=(0, 12), 
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='#F18F01', alpha=0.8, linewidth=1.5))
    
    # Add GPU memory capacity line if available
    if data:
        total_vram_gb = data[0]['gpu_total_vram'] / 1e9
        ax.axhline(y=total_vram_gb, color='red', linestyle='--', linewidth=2, 
                  alpha=0.6, label=f'GPU Capacity: {total_vram_gb:.1f} GB')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add system info
        info = data[0]
        fig.text(0.5, 0.96, 
                f"GPU: {info['gpu_name']} ({total_vram_gb:.1f} GB) | "
                f"dtype: {info['dtype']} | Batch Size: {info['batch_size']}",
                ha='center', fontsize=10, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save plot
    output_path = Path(output_dir) / 'vram_vs_T.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    
    plt.close()


def plot_per_phase_timing(data, output_dir):
    """
    Plot per-phase timing breakdown.
    Shows both absolute times and percentage breakdown.
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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Absolute times (bar chart)
    bars = ax1.bar(labels, times, color=colors, edgecolor='white', linewidth=2)
    
    ax1.set_ylabel('Total Time (ms)', fontweight='bold')
    ax1.set_title(f'Absolute Time Breakdown (T={T_star} tokens)', fontweight='bold', pad=15)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim(bottom=0, top=max(times) * 1.15)
    
    # Add value annotations on bars
    for bar, time, pct in zip(bars, times, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f} ms\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Rotate x labels if needed
    ax1.tick_params(axis='x', rotation=15)
    
    # Plot 2: Percentage breakdown (pie chart)
    explode = [0.05] * len(phases)
    wedges, texts, autotexts = ax2.pie(percentages, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=explode,
                                        wedgeprops=dict(edgecolor='white', linewidth=2))
    
    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax2.set_title(f'Relative Time Distribution (T={T_star} tokens)', 
                 fontweight='bold', pad=15)
    
    # Add system info
    if data:
        info = record
        fig.suptitle(f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | "
                    f"dtype: {info['dtype']} | Total Time: {total_measured:.1f} ms",
                    fontsize=10, y=0.98, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    output_path = Path(output_dir) / 'per_phase_timing.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    
    plt.close()


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
        print("Generating Latency vs T plot...")
        plot_latency_vs_T(results['latency_vs_T'], output_dir)
    
    if 'vram_vs_T' in results:
        print("Generating VRAM vs T plot...")
        plot_vram_vs_T(results['vram_vs_T'], output_dir)
    
    if 'per_phase_timing' in results:
        print("Generating Per-phase Timing plot...")
        plot_per_phase_timing(results['per_phase_timing'], output_dir)
    
    print()
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
