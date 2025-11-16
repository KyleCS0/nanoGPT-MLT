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
from datetime import datetime, timezone

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


def linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, b = np.polyfit(x, y, deg=1)
    return m, b


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


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


def save_metrics_summary(results, output_dir):
    """Write a concise JSON summary with fits and breakdowns."""
    summary = {}

    # Meta
    any_key = next(iter(results)) if results else None
    first_list = results.get(any_key, [{}])
    meta_src = first_list[0] if first_list else {}
    meta = {
        'gpu': meta_src.get('gpu_name'),
        'gpu_total_vram_bytes': meta_src.get('gpu_total_vram'),
        'python_version': meta_src.get('python_version'),
        'pytorch_version': meta_src.get('pytorch_version'),
        'cuda_version': meta_src.get('cuda_version'),
        'dtype': meta_src.get('dtype'),
        'batch_size': meta_src.get('batch_size'),
        'prompt_length': meta_src.get('prompt_length'),
        'model_config': meta_src.get('model_config'),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }

    # Latency section
    if 'latency_vs_T' in results and results['latency_vs_T']:
        lat = sorted(results['latency_vs_T'], key=lambda d: d['T'])
        T_vals = [int(d['T']) for d in lat]
        total_ms = [float(d['time_total_ms_median']) for d in lat]
        per_tok_ms = [float(d['time_per_token_ms_median']) for d in lat]

        # Fits
        a, b, c = quadratic_fit(T_vals, total_ms)
        T_dense = np.asarray(T_vals, dtype=float)
        total_fit = a * T_dense ** 2 + b * T_dense + c
        r2_total = r2_score(total_ms, total_fit)

        m_pt, b_pt = linear_fit(T_vals, per_tok_ms)
        per_tok_fit = m_pt * T_dense + b_pt
        r2_pt = r2_score(per_tok_ms, per_tok_fit)

        summary['latency'] = {
            'points': [
                {'T': int(T_vals[i]),
                 'total_ms': float(total_ms[i]),
                 'per_token_ms': float(per_tok_ms[i])}
                for i in range(len(T_vals))
            ],
            'fit_total_ms_vs_T': {
                'model': 'quadratic',
                'coeffs': {'a': float(a), 'b': float(b), 'c': float(c)},
                'r2': float(r2_total),
                'T_range': [int(min(T_vals)), int(max(T_vals))],
            },
            'fit_per_token_ms_vs_T': {
                'model': 'linear',
                'coeffs': {'m': float(m_pt), 'b': float(b_pt)},
                'r2': float(r2_pt),
                'T_range': [int(min(T_vals)), int(max(T_vals))],
            },
        }
        meta['T_values'] = T_vals

    # VRAM section
    if 'vram_vs_T' in results and results['vram_vs_T']:
        vram = sorted(results['vram_vs_T'], key=lambda d: d['T'])
        T_vals_v = [int(d['T']) for d in vram]
        abs_MB = [float(d['peak_memory_bytes']) / 1e6 for d in vram]
        baseline = abs_MB[0]
        delta_MB = [float(v - baseline) for v in abs_MB]

        # Linear fit of delta vs T
        m_v, b_v = linear_fit(T_vals_v, delta_MB)
        y_pred_v = (np.asarray(T_vals_v, dtype=float) * m_v + b_v)
        r2_v = r2_score(delta_MB, y_pred_v)

        summary['vram'] = {
            'unit': 'MB',
            'baseline_T': int(T_vals_v[0]),
            'points': [
                {'T': int(T_vals_v[i]), 'abs_MB': float(abs_MB[i]), 'delta_MB': float(delta_MB[i])}
                for i in range(len(T_vals_v))
            ],
            'fit_delta_MB_vs_T': {
                'model': 'linear',
                'coeffs': {'m': float(m_v), 'b': float(b_v)},
                'r2': float(r2_v),
                'T_range': [int(min(T_vals_v)), int(max(T_vals_v))],
            },
        }

    # Per-phase section
    if 'per_phase_timing' in results and results['per_phase_timing']:
        rec = results['per_phase_timing'][-1]
        phases = dict(rec.get('time_phase_ms', {}))
        total = float(phases.get('total', 0.0))
        # Percentages relative to total if available
        percent = {}
        for k, v in phases.items():
            if k == 'total':
                continue
            pct = (float(v) / total) if total > 0 else 0.0
            percent[k] = float(pct)

        summary['per_phase'] = {
            'T_star': int(rec.get('T_star', 0)),
            'times_ms': {k: float(v) for k, v in phases.items()},
            'percent': percent,
        }

    summary['meta'] = meta

    # Save
    out_path = Path(output_dir) / 'metrics_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


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

    # Always emit JSON metrics summary
    print("Writing metrics summary JSON...")
    save_metrics_summary(results, output_dir)
    
    print()
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
