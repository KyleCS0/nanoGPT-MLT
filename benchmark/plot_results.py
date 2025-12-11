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
    """Load and parse benchmark results from JSONL file.

    Groups results by (benchmark_name, version) for version-aware plotting.
    Legacy results without version field are grouped under 'legacy'.
    """
    results = defaultdict(list)

    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                benchmark_name = record['benchmark_name']
                version = record.get('version', 'legacy')
                key = (benchmark_name, version)
                results[key].append(record)

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


def get_cache_label(version):
    """Convert version to human-readable cache label."""
    if version == 'v0':
        return 'KV-cache OFF'
    elif version == 'v1':
        return 'KV-cache ON'
    elif version == 'legacy':
        return ''
    else:
        return f'KV-cache {version}'


def plot_latency_vs_T(data, output_dir, version='legacy'):
    """
    Plot Latency vs T benchmark results.
    Creates separate standalone plots for total time and per-token time.

    Args:
        data: List of benchmark records
        output_dir: Directory to save plots
        version: Version string for filename (e.g., 'v0', 'v1', 'legacy')
    """
    # Sort by T value
    data = sorted(data, key=lambda x: x['T'])

    T_values = [d['T'] for d in data]
    time_total_median = [d['time_total_ms_median'] for d in data]
    time_total_std = [d['time_total_ms_std'] for d in data]
    time_per_token = [d['time_per_token_ms_median'] for d in data]

    # Human-readable label for titles
    cache_label = get_cache_label(version)
    title_suffix = f" ({cache_label})" if cache_label else ""
    # Version suffix for filenames
    version_suffix = f"_{version}" if version != 'legacy' else ""

    # Plot 1: Total Time vs T (standalone)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    ax1.errorbar(T_values, time_total_median, yerr=time_total_std, fmt='o',
                 markersize=4.5, elinewidth=1.0, capsize=2,
                 color='black', ecolor='#888888', label='Median +/- std')
    # Quadratic fit to highlight scaling
    a, b, c = quadratic_fit(T_values, time_total_median)
    T_dense = np.linspace(min(T_values), max(T_values), 256)
    ax1.plot(T_dense, a*T_dense**2 + b*T_dense + c, '-', color='#2E86AB', linewidth=2,
             label='Quadratic fit')

    ax1.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax1.set_ylabel('Total Generation Time (ms)', fontweight='bold')
    ax1.set_title(f'Total Latency vs Generation Length{title_suffix}', fontweight='bold', pad=15)
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
    output_path1 = Path(output_dir) / f'latency_total_vs_T{version_suffix}.png'
    plt.savefig(output_path1, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path1}")
    plt.close()

    # Plot 2: Time per Token vs T (standalone)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    ax2.plot(T_values, time_per_token, 's-', linewidth=1.8, markersize=4.5,
             color='#A23B72')

    ax2.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax2.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax2.set_title(f'Per-Token Latency vs Generation Length{title_suffix}', fontweight='bold', pad=15)
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
    output_path2 = Path(output_dir) / f'latency_per_token_vs_T{version_suffix}.png'
    plt.savefig(output_path2, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path2}")
    plt.close()


def plot_vram_vs_T(data, output_dir, version='legacy'):
    """
    Plot VRAM usage vs T: relative growth from first T point (in MB).

    Args:
        data: List of benchmark records
        output_dir: Directory to save plots
        version: Version string for filename (e.g., 'v0', 'v1', 'legacy')
    """
    # Sort by T value
    data = sorted(data, key=lambda x: x['T'])

    T_values = [d['T'] for d in data]
    peak_memory_mb = [d['peak_memory_bytes'] / 1e6 for d in data]

    # Baseline subtract: use first T point as baseline
    baseline_memory_mb = peak_memory_mb[0]
    relative_memory_mb = [mem - baseline_memory_mb for mem in peak_memory_mb]

    # Human-readable label for titles
    cache_label = get_cache_label(version)
    title_suffix = f" ({cache_label})" if cache_label else ""
    # Version suffix for filenames
    version_suffix = f"_{version}" if version != 'legacy' else ""

    # Create standalone figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(T_values, relative_memory_mb, 'o-', linewidth=1.8, markersize=4.5,
            color='#2E86AB')
    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Additional Memory Usage (MB)', fontweight='bold')
    ax.set_title(f'GPU Memory Growth vs Generation Length{title_suffix}', fontweight='bold', pad=15)
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
    output_path = Path(output_dir) / f'vram_vs_T{version_suffix}.png'
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
    """Write a concise JSON summary with fits and breakdowns.

    Results are keyed by (benchmark_name, version) tuples.
    """
    summary = {}

    # Meta - get from first available result
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

    # Process each (benchmark_name, version) combination
    for (benchmark_name, version), data in results.items():
        version_key = version if version != 'legacy' else 'default'

        if benchmark_name == 'latency_vs_T' and data:
            lat = sorted(data, key=lambda d: d['T'])
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

            latency_data = {
                'version': version,
                'cache_label': get_cache_label(version),
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

            if 'latency' not in summary:
                summary['latency'] = {}
            summary['latency'][version_key] = latency_data
            meta['T_values'] = T_vals

        elif benchmark_name == 'vram_vs_T' and data:
            vram = sorted(data, key=lambda d: d['T'])
            T_vals_v = [int(d['T']) for d in vram]
            abs_MB = [float(d['peak_memory_bytes']) / 1e6 for d in vram]
            baseline = abs_MB[0]
            delta_MB = [float(v - baseline) for v in abs_MB]

            # Linear fit of delta vs T
            m_v, b_v = linear_fit(T_vals_v, delta_MB)
            y_pred_v = (np.asarray(T_vals_v, dtype=float) * m_v + b_v)
            r2_v = r2_score(delta_MB, y_pred_v)

            vram_data = {
                'version': version,
                'cache_label': get_cache_label(version),
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

            if 'vram' not in summary:
                summary['vram'] = {}
            summary['vram'][version_key] = vram_data

        elif benchmark_name == 'per_phase_timing' and data:
            rec = data[-1]
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


# --- Comparison plots: multiple versions on the same graph ---

# Color palette for version comparison
VERSION_COLORS = {
    'v0': '#E63946',  # Red for no-cache
    'v1': '#2A9D8F',  # Teal for with-cache
    'legacy': '#457B9D',  # Blue for legacy
}

VERSION_MARKERS = {
    'v0': 'o',
    'v1': 's',
    'legacy': '^',
}


def plot_latency_comparison(results, output_dir):
    """
    Plot latency comparison: v0 vs v1 on the same graph.
    Creates:
      1. Total latency comparison
      2. Per-token latency comparison (best for showing KV-cache advantage)
      3. Speedup ratio plot
    """
    # Collect latency data for all versions
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if len(latency_data) < 2:
        print("  Need at least 2 versions for comparison plots. Skipping.")
        return

    # Get system info from first available data
    first_data = next(iter(latency_data.values()))[0]

    # --- Plot 1: Total Latency Comparison ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for version, data in sorted(latency_data.items()):
        T_values = [d['T'] for d in data]
        time_total = [d['time_total_ms_median'] for d in data]
        time_std = [d['time_total_ms_std'] for d in data]
        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax1.errorbar(T_values, time_total, yerr=time_std, fmt=f'{marker}-',
                     markersize=5, linewidth=1.8, capsize=2,
                     color=color, ecolor=color, alpha=0.8,
                     label=label)

        # Add fit line
        a, b, c = quadratic_fit(T_values, time_total)
        T_dense = np.linspace(min(T_values), max(T_values), 256)
        ax1.plot(T_dense, a*T_dense**2 + b*T_dense + c, '--',
                 color=color, alpha=0.5, linewidth=1.5)

    ax1.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax1.set_ylabel('Total Generation Time (ms)', fontweight='bold')
    ax1.set_title('Total Latency: KV-Cache Comparison', fontweight='bold', pad=15)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.3)

    fig1.text(0.5, 0.02,
              f"GPU: {first_data['gpu_name']} | PyTorch {first_data['pytorch_version']} | "
              f"dtype: {first_data['dtype']} | Batch Size: {first_data['batch_size']}",
              ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path1 = Path(output_dir) / 'comparison_total_latency.png'
    plt.savefig(path1, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path1}")
    plt.close()

    # --- Plot 2: Per-Token Latency Comparison (Most Important!) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for version, data in sorted(latency_data.items()):
        T_values = [d['T'] for d in data]
        time_per_token = [d['time_per_token_ms_median'] for d in data]
        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax2.plot(T_values, time_per_token, f'{marker}-',
                 markersize=6, linewidth=2,
                 color=color, label=label)

    ax2.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax2.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax2.set_title('Per-Token Latency: KV-Cache Comparison', fontweight='bold', pad=15)
    ax2.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Add annotation explaining the key insight
    ax2.annotate('KV-cache: O(1) per token\n(flat scaling)',
                 xy=(800, 1.34), fontsize=9, color=VERSION_COLORS['v1'],
                 ha='center', style='italic')
    ax2.annotate('No cache: O(T) per token\n(linear growth)',
                 xy=(800, 1.52), fontsize=9, color=VERSION_COLORS['v0'],
                 ha='center', style='italic')

    fig2.text(0.5, 0.02,
              f"GPU: {first_data['gpu_name']} | PyTorch {first_data['pytorch_version']} | "
              f"dtype: {first_data['dtype']} | Batch Size: {first_data['batch_size']}",
              ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path2 = Path(output_dir) / 'comparison_per_token_latency.png'
    plt.savefig(path2, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path2}")
    plt.close()

    # --- Plot 3: Speedup Ratio (v0/v1) ---
    if 'v0' in latency_data and 'v1' in latency_data:
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        v0_data = {d['T']: d['time_total_ms_median'] for d in latency_data['v0']}
        v1_data = {d['T']: d['time_total_ms_median'] for d in latency_data['v1']}

        common_T = sorted(set(v0_data.keys()) & set(v1_data.keys()))
        speedup = [v0_data[T] / v1_data[T] for T in common_T]

        ax3.plot(common_T, speedup, 'o-', markersize=7, linewidth=2.5,
                 color='#6A4C93', label='Speedup (v0/v1)')
        ax3.axhline(y=1.0, color='#888888', linestyle='--', linewidth=1.5,
                    label='Breakeven (1.0x)')

        # Fill regions
        ax3.fill_between(common_T, speedup, 1.0,
                         where=[s > 1 for s in speedup],
                         alpha=0.3, color='#2A9D8F', label='KV-cache faster')
        ax3.fill_between(common_T, speedup, 1.0,
                         where=[s < 1 for s in speedup],
                         alpha=0.3, color='#E63946', label='KV-cache slower')

        ax3.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
        ax3.set_ylabel('Speedup Ratio', fontweight='bold')
        ax3.set_title('KV-Cache Speedup vs Generation Length', fontweight='bold', pad=15)
        ax3.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.3)

        # Find crossover point
        crossover_T = None
        for i, T in enumerate(common_T[:-1]):
            if speedup[i] < 1.0 and speedup[i+1] >= 1.0:
                # Linear interpolation
                crossover_T = T + (common_T[i+1] - T) * (1.0 - speedup[i]) / (speedup[i+1] - speedup[i])
                break

        if crossover_T:
            ax3.axvline(x=crossover_T, color='#F4A261', linestyle=':', linewidth=2)
            ax3.annotate(f'Crossover: T≈{crossover_T:.0f}',
                         xy=(crossover_T, 1.0), xytext=(crossover_T + 50, 1.08),
                         fontsize=10, fontweight='bold', color='#F4A261',
                         arrowprops=dict(arrowstyle='->', color='#F4A261'))

        fig3.text(0.5, 0.02,
                  f"GPU: {first_data['gpu_name']} | PyTorch {first_data['pytorch_version']} | "
                  f"dtype: {first_data['dtype']} | Speedup > 1.0 means KV-cache is faster",
                  ha='center', fontsize=9, style='italic', color='#555555')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path3 = Path(output_dir) / 'comparison_speedup.png'
        plt.savefig(path3, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  Saved: {path3}")
        plt.close()

    # --- Plot 4: Extrapolated Projection ---
    if 'v0' in latency_data and 'v1' in latency_data:
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        # Get fits
        v0_T = [d['T'] for d in latency_data['v0']]
        v0_time = [d['time_total_ms_median'] for d in latency_data['v0']]
        v1_T = [d['T'] for d in latency_data['v1']]
        v1_time = [d['time_total_ms_median'] for d in latency_data['v1']]

        a0, b0, c0 = quadratic_fit(v0_T, v0_time)
        a1, b1, c1 = quadratic_fit(v1_T, v1_time)

        # Extrapolate to 2x max T
        max_T = max(max(v0_T), max(v1_T))
        T_extrap = np.linspace(min(min(v0_T), min(v1_T)), max_T * 2, 512)

        v0_extrap = a0 * T_extrap**2 + b0 * T_extrap + c0
        v1_extrap = a1 * T_extrap**2 + b1 * T_extrap + c1

        # Plot actual data points
        ax4.scatter(v0_T, v0_time, s=40, color=VERSION_COLORS['v0'],
                    marker='o', zorder=5, label='KV-cache OFF (measured)')
        ax4.scatter(v1_T, v1_time, s=40, color=VERSION_COLORS['v1'],
                    marker='s', zorder=5, label='KV-cache ON (measured)')

        # Plot extrapolated fits
        ax4.plot(T_extrap, v0_extrap, '-', color=VERSION_COLORS['v0'],
                 linewidth=2, alpha=0.7, label='KV-cache OFF (projected)')
        ax4.plot(T_extrap, v1_extrap, '-', color=VERSION_COLORS['v1'],
                 linewidth=2, alpha=0.7, label='KV-cache ON (projected)')

        # Add vertical line at max measured T
        ax4.axvline(x=max_T, color='#888888', linestyle='--', linewidth=1,
                    label=f'Measured limit (T={max_T})')

        # Annotate speedup at extrapolated points
        extrap_T = max_T * 2
        v0_at_extrap = a0 * extrap_T**2 + b0 * extrap_T + c0
        v1_at_extrap = a1 * extrap_T**2 + b1 * extrap_T + c1
        speedup_extrap = v0_at_extrap / v1_at_extrap

        ax4.annotate(f'At T={int(extrap_T)}: {speedup_extrap:.1f}x speedup',
                     xy=(extrap_T, v1_at_extrap), xytext=(extrap_T - 200, v1_at_extrap + 500),
                     fontsize=10, fontweight='bold', color=VERSION_COLORS['v1'],
                     arrowprops=dict(arrowstyle='->', color=VERSION_COLORS['v1']))

        ax4.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
        ax4.set_ylabel('Total Generation Time (ms)', fontweight='bold')
        ax4.set_title('Latency Projection: KV-Cache Scaling Advantage', fontweight='bold', pad=15)
        ax4.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
        ax4.grid(True, linestyle='--', alpha=0.3)

        fig4.text(0.5, 0.02,
                  f"GPU: {first_data['gpu_name']} | Extrapolation based on quadratic fit | "
                  f"v0: O(T²), v1: ~O(T)",
                  ha='center', fontsize=9, style='italic', color='#555555')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path4 = Path(output_dir) / 'comparison_extrapolation.png'
        plt.savefig(path4, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  Saved: {path4}")
        plt.close()


def plot_vram_comparison(results, output_dir):
    """
    Plot VRAM comparison: v0 vs v1 on the same graph.
    """
    # Collect VRAM data for all versions
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'vram_vs_T':
            vram_data[version] = sorted(data, key=lambda x: x['T'])

    if len(vram_data) < 2:
        print("  Need at least 2 versions for VRAM comparison. Skipping.")
        return

    first_data = next(iter(vram_data.values()))[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for version, data in sorted(vram_data.items()):
        T_values = [d['T'] for d in data]
        peak_mb = [d['peak_memory_bytes'] / 1e6 for d in data]
        baseline = peak_mb[0]
        delta_mb = [m - baseline for m in peak_mb]

        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(T_values, delta_mb, f'{marker}-',
                markersize=6, linewidth=2,
                color=color, label=label)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Additional Memory Usage (MB)', fontweight='bold')
    ax.set_title('Memory Growth: KV-Cache Comparison', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | KV-cache requires O(T) additional memory",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_vram.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
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

    # Group keys by benchmark type
    benchmark_versions = defaultdict(list)
    for (benchmark_name, version) in results.keys():
        benchmark_versions[benchmark_name].append(version)

    print(f"Found {len(results)} benchmark/version combinations:")
    for bname, versions in benchmark_versions.items():
        print(f"  {bname}: {versions}")
    print()

    # Generate plots for each benchmark type and version
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            cache_label = get_cache_label(version)
            label_str = f" ({cache_label})" if cache_label else ""
            print(f"Generating Latency vs T plots{label_str}...")
            plot_latency_vs_T(data, output_dir, version=version)

        elif benchmark_name == 'vram_vs_T':
            cache_label = get_cache_label(version)
            label_str = f" ({cache_label})" if cache_label else ""
            print(f"Generating VRAM vs T plot{label_str}...")
            plot_vram_vs_T(data, output_dir, version=version)

        elif benchmark_name == 'per_phase_timing':
            print("Generating Per-phase Timing plots...")
            plot_per_phase_timing(data, output_dir)

    # Always emit JSON metrics summary
    print("Writing metrics summary JSON...")
    save_metrics_summary(results, output_dir)

    # Generate comparison plots if multiple versions exist
    print("\nGenerating comparison plots...")
    plot_latency_comparison(results, output_dir)
    plot_vram_comparison(results, output_dir)

    print()
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
