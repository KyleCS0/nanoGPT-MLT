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

    IMPORTANT: Deduplicates by taking the FIRST record for each unique
    (benchmark_name, version, T) combination to handle multiple runs.
    Rationale: First complete run is usually consistent; later partial runs may corrupt data.
    """
    # First pass: collect all records
    raw_results = defaultdict(list)
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                benchmark_name = record['benchmark_name']
                version = record.get('version', 'legacy')
                key = (benchmark_name, version)
                raw_results[key].append(record)

    # Second pass: deduplicate by T value (keep FIRST occurrence)
    # Rationale: The first complete benchmark run is usually consistent.
    # Later runs may be partial/interrupted, causing inconsistent data.
    results = defaultdict(list)
    for key, records in raw_results.items():
        benchmark_name = key[0]
        if benchmark_name in ('latency_vs_T', 'vram_vs_T'):
            # Deduplicate by T - keep FIRST record for each T
            by_T = {}
            for r in records:
                T = r.get('T')
                if T is not None and T not in by_T:
                    by_T[T] = r  # Only keep if not already seen
            results[key] = list(by_T.values())
        else:
            # For other benchmarks, keep all records
            results[key] = records

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
    labels = {
        'v0': 'v0: No cache',
        'v1': 'v1: KV-cache',
        'v2': 'v2: KV + INT8',
        'v3': 'v3: KV + cross-layer',
        'v4': 'v4: KV + INT8 + cross-layer',
        'legacy': '',
    }
    return labels.get(version, f'{version}')


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
    Plot per-phase timing breakdown for all versions.
    Creates a grouped bar chart comparing phases across versions.
    """
    if not data:
        print("  No per-phase timing data found.")
        return

    # Group data by version, taking the latest record per version
    version_data = {}
    for record in data:
        version = record.get('version', 'unknown')
        version_data[version] = record  # Latest wins

    if not version_data:
        print("  No per-phase timing data found.")
        return

    # Sort versions for consistent ordering (v0, v1, v2, ...)
    versions = sorted(version_data.keys())
    T_star = version_data[versions[0]]['T_star']

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

    # Get phases that exist in any version
    all_phases = set()
    for v in versions:
        all_phases.update(version_data[v]['time_phase_ms'].keys())
    phases = [p for p in phase_order if p in all_phases and p != 'total']

    # --- Grouped bar chart comparing versions ---
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(phases))
    width = 0.8 / len(versions)  # Width of each bar
    version_colors = plt.cm.tab10(np.linspace(0, 1, len(versions)))

    for i, version in enumerate(versions):
        phase_times = version_data[version]['time_phase_ms']
        times = [phase_times.get(p, 0) for p in phases]
        desc = version_data[version].get('version_description', version)
        offset = (i - len(versions)/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=f'{version} ({desc})',
                     color=version_colors[i], edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{t:.0f}', ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_title(f'Per-Phase Timing Breakdown by Version (T={T_star})', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([phase_labels[p] for p in phases])
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add system info from first record
    info = version_data[versions[0]]
    fig.text(0.5, 0.02,
            f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | dtype: {info['dtype']}",
            ha='center', fontsize=9, style='italic', color='#555555')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    bars_path = Path(output_dir) / 'per_phase_breakdown_bar.png'
    fig.savefig(bars_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {bars_path}")
    plt.close(fig)

    # --- Stacked bar chart showing total time breakdown ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(versions))
    for phase in phases:
        times = [version_data[v]['time_phase_ms'].get(phase, 0) for v in versions]
        ax2.bar(versions, times, bottom=bottom, label=phase_labels[phase],
               color=phase_colors[phase], edgecolor='white', linewidth=0.5)
        bottom += np.array(times)

    # Add total time labels on top
    for i, v in enumerate(versions):
        total = version_data[v]['time_phase_ms'].get('total', sum(
            version_data[v]['time_phase_ms'].get(p, 0) for p in phases))
        ax2.text(i, bottom[i] + 10, f'{total:.0f} ms', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax2.set_ylabel('Time (ms)', fontweight='bold')
    ax2.set_xlabel('Version', fontweight='bold')
    ax2.set_title(f'Total Time Breakdown by Phase (T={T_star})', fontweight='bold', pad=15)
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig2.text(0.5, 0.02,
             f"GPU: {info['gpu_name']} | PyTorch {info['pytorch_version']} | dtype: {info['dtype']}",
             ha='center', fontsize=9, style='italic', color='#555555')

    fig2.tight_layout(rect=[0, 0.04, 1, 1])
    stacked_path = Path(output_dir) / 'per_phase_breakdown_stacked.png'
    fig2.savefig(stacked_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {stacked_path}")
    plt.close(fig2)


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
            # Group by version
            version_data = {}
            for rec in data:
                version = rec.get('version', 'unknown')
                version_data[version] = rec

            if 'per_phase' not in summary:
                summary['per_phase'] = {}

            for version, rec in version_data.items():
                phases = dict(rec.get('time_phase_ms', {}))
                total = float(phases.get('total', 0.0))
                percent = {}
                for k, v in phases.items():
                    if k == 'total':
                        continue
                    pct = (float(v) / total) if total > 0 else 0.0
                    percent[k] = float(pct)

                summary['per_phase'][version] = {
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

def compute_ci95(std, n_samples=10):
    """Compute 95% confidence interval half-width from standard deviation.

    Uses t-distribution approximation for small samples.
    CI = mean ± t_{0.975, n-1} * std / sqrt(n)
    For n=10, t_{0.975,9} ≈ 2.262
    """
    from scipy import stats
    if n_samples <= 1:
        return std
    t_val = stats.t.ppf(0.975, df=n_samples - 1)
    return t_val * std / np.sqrt(n_samples)


# Color palette for version comparison
VERSION_COLORS = {
    'v0': '#E63946',  # Red for no-cache
    'v1': '#2A9D8F',  # Teal for with-cache
    'v2': '#F4A261',  # Orange for INT8 quantization
    'v3': '#6A4C93',  # Purple for cross-layer sharing
    'v4': '#264653',  # Dark teal for combined
    'legacy': '#457B9D',  # Blue for legacy
}

VERSION_MARKERS = {
    'v0': 'o',
    'v1': 's',
    'v2': '^',
    'v3': 'd',
    'v4': 'p',
    'legacy': 'x',
}

VERSION_LABELS = {
    'v0': 'v0: No cache',
    'v1': 'v1: KV-cache',
    'v2': 'v2: KV + INT8',
    'v3': 'v3: KV + cross-layer',
    'v4': 'v4: KV + INT8 + cross-layer',
    'legacy': 'Legacy',
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


def plot_throughput_comparison(results, output_dir):
    """
    Plot throughput comparison: tokens/second vs T for all versions.
    Throughput = 1000 / time_per_token_ms (tokens/sec)
    """
    # Collect latency data for all versions
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if len(latency_data) < 2:
        print("  Need at least 2 versions for throughput comparison. Skipping.")
        return

    first_data = next(iter(latency_data.values()))[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for version, data in sorted(latency_data.items()):
        T_values = [d['T'] for d in data]
        # Throughput = 1000 / time_per_token_ms (tokens/second)
        throughput = [1000.0 / d['time_per_token_ms_median'] for d in data]
        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(T_values, throughput, f'{marker}-',
                markersize=6, linewidth=2,
                color=color, label=label)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Throughput (tokens/second)', fontweight='bold')
    ax.set_title('Generation Throughput: KV-Cache Comparison', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add annotation
    ax.annotate('Higher is better',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=9, ha='right', va='top', style='italic', color='#555555')

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Throughput = 1000 / time_per_token_ms",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_throughput.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_memory_efficiency_comparison(results, output_dir):
    """
    Plot memory efficiency: latency per MB of VRAM used.
    Lower is better (faster per unit memory).
    """
    # Collect latency and vram data for all versions
    latency_data = {}
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = {d['T']: d for d in data}
        elif benchmark_name == 'vram_vs_T':
            vram_data[version] = {d['T']: d for d in data}

    # Find versions that have both latency and vram data
    common_versions = set(latency_data.keys()) & set(vram_data.keys())
    if len(common_versions) < 2:
        print("  Need at least 2 versions with both latency and VRAM data. Skipping memory efficiency plot.")
        return

    first_version = sorted(common_versions)[0]
    first_data = list(latency_data[first_version].values())[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for version in sorted(common_versions):
        lat_by_T = latency_data[version]
        vram_by_T = vram_data[version]

        # Find common T values
        common_T = sorted(set(lat_by_T.keys()) & set(vram_by_T.keys()))
        if not common_T:
            continue

        T_values = []
        efficiency = []  # ms per MB

        for T in common_T:
            lat = lat_by_T[T]['time_total_ms_median']
            vram_mb = vram_by_T[T]['peak_memory_bytes'] / 1e6
            if vram_mb > 0:
                T_values.append(T)
                efficiency.append(lat / vram_mb)  # ms per MB

        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(T_values, efficiency, f'{marker}-',
                markersize=6, linewidth=2,
                color=color, label=label)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Latency per MB VRAM (ms/MB)', fontweight='bold')
    ax.set_title('Memory Efficiency: Latency per Memory Used', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add annotation
    ax.annotate('Lower is better\n(more efficient)',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=9, ha='right', va='top', style='italic', color='#555555')

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Efficiency = total_latency_ms / peak_vram_MB",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_memory_efficiency.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_latency_comparison_with_ci(results, output_dir, default_n_samples=10):
    """
    Plot per-token latency comparison with 95% confidence intervals.

    Uses n_samples from data if available, otherwise falls back to default.
    """
    # Collect latency data for all versions
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if len(latency_data) < 2:
        print("  Need at least 2 versions for CI comparison. Skipping.")
        return

    first_data = next(iter(latency_data.values()))[0]
    # Get n_samples from data if available
    n_samples = first_data.get('n_samples', default_n_samples)

    fig, ax = plt.subplots(figsize=(10, 6))

    for version, data in sorted(latency_data.items()):
        T_values = np.array([d['T'] for d in data])
        time_per_token = np.array([d['time_per_token_ms_median'] for d in data])

        # Compute 95% CI from std, using per-point n_samples if available
        time_std = np.array([d.get('time_per_token_ms_std', 0) for d in data])
        n_per_point = np.array([d.get('n_samples', default_n_samples) for d in data])
        ci95 = np.array([compute_ci95(s, n) for s, n in zip(time_std, n_per_point)])

        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        # Plot line with markers
        ax.plot(T_values, time_per_token, f'{marker}-',
                markersize=5, linewidth=2,
                color=color, label=label)

        # Add CI band
        ax.fill_between(T_values,
                        time_per_token - ci95,
                        time_per_token + ci95,
                        color=color, alpha=0.2)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax.set_title('Per-Token Latency with 95% Confidence Intervals', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Shaded regions: 95% CI (n={n_samples})",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_per_token_latency_ci.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_total_latency_comparison_with_ci(results, output_dir, default_n_samples=10):
    """
    Plot total latency comparison with 95% confidence intervals.

    Uses n_samples from data if available, otherwise falls back to default.
    """
    # Collect latency data for all versions
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if len(latency_data) < 2:
        print("  Need at least 2 versions for CI comparison. Skipping.")
        return

    first_data = next(iter(latency_data.values()))[0]
    # Get n_samples from data if available
    n_samples = first_data.get('n_samples', default_n_samples)

    fig, ax = plt.subplots(figsize=(10, 6))

    for version, data in sorted(latency_data.items()):
        T_values = np.array([d['T'] for d in data])
        time_total = np.array([d['time_total_ms_median'] for d in data])

        # Compute 95% CI from std, using per-point n_samples if available
        time_std = np.array([d.get('time_total_ms_std', 0) for d in data])
        n_per_point = np.array([d.get('n_samples', default_n_samples) for d in data])
        ci95 = np.array([compute_ci95(s, n) for s, n in zip(time_std, n_per_point)])

        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        # Plot line with markers
        ax.plot(T_values, time_total, f'{marker}-',
                markersize=5, linewidth=2,
                color=color, label=label)

        # Add CI band
        ax.fill_between(T_values,
                        time_total - ci95,
                        time_total + ci95,
                        color=color, alpha=0.2)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Total Generation Time (ms)', fontweight='bold')
    ax.set_title('Total Latency with 95% Confidence Intervals', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Shaded regions: 95% CI (n={n_samples})",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_total_latency_ci.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_pareto_memory_latency(results, output_dir, T_target=512):
    """
    Plot Memory vs Latency Pareto chart.
    X: Peak VRAM (MB), Y: Total Latency (ms) at a fixed T value.
    Each point represents one version - shows tradeoff between memory and speed.
    """
    latency_data = {}
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = {d['T']: d for d in data}
        elif benchmark_name == 'vram_vs_T':
            vram_data[version] = {d['T']: d for d in data}

    # Find versions with both latency and VRAM data at T_target
    versions_with_data = []
    for v in set(latency_data.keys()) & set(vram_data.keys()):
        if T_target in latency_data[v] and T_target in vram_data[v]:
            versions_with_data.append(v)

    if len(versions_with_data) < 2:
        print(f"  Need at least 2 versions with data at T={T_target} for Pareto plot. Skipping.")
        return

    first_data = list(latency_data[versions_with_data[0]].values())[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for version in sorted(versions_with_data):
        vram_mb = vram_data[version][T_target]['peak_memory_bytes'] / 1e6
        latency_ms = latency_data[version][T_target]['time_total_ms_median']

        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.scatter(vram_mb, latency_ms, s=150, c=color, marker=marker,
                   label=label, edgecolors='white', linewidth=1.5, zorder=5)

        # Add version label next to point
        ax.annotate(version, (vram_mb, latency_ms),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xlabel('Peak VRAM Usage (MB)', fontweight='bold')
    ax.set_ylabel(f'Total Latency at T={T_target} (ms)', fontweight='bold')
    ax.set_title(f'Memory-Latency Tradeoff (T={T_target})', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add Pareto frontier annotation
    ax.annotate('← Lower is better (both axes)',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, ha='left', va='bottom', style='italic', color='#555555')

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Ideal: bottom-left corner",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'pareto_memory_latency.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_throughput_comparison_with_ci(results, output_dir, default_n_samples=10):
    """
    Plot throughput comparison with 95% confidence intervals.
    Throughput = 1000 / time_per_token_ms (tokens/sec)
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if len(latency_data) < 2:
        print("  Need at least 2 versions for throughput CI comparison. Skipping.")
        return

    first_data = next(iter(latency_data.values()))[0]
    n_samples = first_data.get('n_samples', default_n_samples)

    fig, ax = plt.subplots(figsize=(10, 6))

    for version, data in sorted(latency_data.items()):
        T_values = np.array([d['T'] for d in data])
        time_per_token = np.array([d['time_per_token_ms_median'] for d in data])
        throughput = 1000.0 / time_per_token  # tokens/sec

        # Propagate uncertainty: if throughput = 1000/t, then
        # delta_throughput ≈ throughput * (delta_t / t)
        time_std = np.array([d.get('time_per_token_ms_std', 0) for d in data])
        n_per_point = np.array([d.get('n_samples', default_n_samples) for d in data])
        ci95_time = np.array([compute_ci95(s, n) for s, n in zip(time_std, n_per_point)])

        # CI for throughput (error propagation)
        ci95_throughput = throughput * (ci95_time / time_per_token)

        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(T_values, throughput, f'{marker}-',
                markersize=5, linewidth=2,
                color=color, label=label)

        ax.fill_between(T_values,
                        throughput - ci95_throughput,
                        throughput + ci95_throughput,
                        color=color, alpha=0.2)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Throughput (tokens/second)', fontweight='bold')
    ax.set_title('Generation Throughput with 95% Confidence Intervals', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.annotate('Higher is better',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=9, ha='right', va='top', style='italic', color='#555555')

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Shaded regions: 95% CI (n={n_samples})",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_throughput_ci.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_perplexity_comparison(results, output_dir):
    """
    Plot perplexity comparison across versions as a bar chart.

    Perplexity is measured via forward pass (teacher forcing), so all versions
    should produce identical results. This plot verifies that optimizations
    don't degrade model quality.
    """
    # Collect perplexity data
    perplexity_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'perplexity':
            # Take the latest record for each version
            if data:
                perplexity_data[version] = data[-1]

    if not perplexity_data:
        print("  No perplexity data found. Skipping perplexity plot.")
        return

    # Sort versions
    versions = sorted(perplexity_data.keys())
    perplexities = [perplexity_data[v]['perplexity'] for v in versions]

    # Get baseline (v0) for comparison
    baseline_ppl = perplexity_data.get('v0', {}).get('perplexity', perplexities[0])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    x_pos = np.arange(len(versions))
    colors = [VERSION_COLORS.get(v, '#666666') for v in versions]

    bars = ax.bar(x_pos, perplexities, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, ppl) in enumerate(zip(bars, perplexities)):
        height = bar.get_height()
        # Calculate degradation from baseline
        if baseline_ppl > 0:
            deg = ((ppl - baseline_ppl) / baseline_ppl) * 100
            deg_str = f"\n({deg:+.2f}%)" if versions[i] != 'v0' else "\n(baseline)"
        else:
            deg_str = ""
        ax.annotate(f'{ppl:.2f}{deg_str}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Labels
    version_labels = [get_cache_label(v) for v in versions]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(version_labels, rotation=15, ha='right')

    ax.set_xlabel('Version', fontweight='bold')
    ax.set_ylabel('Perplexity', fontweight='bold')
    ax.set_title('Perplexity Comparison Across Versions (WikiText-2)', fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Set y-axis to start near minimum value for better visualization
    min_ppl = min(perplexities)
    max_ppl = max(perplexities)
    margin = (max_ppl - min_ppl) * 0.3 if max_ppl != min_ppl else min_ppl * 0.1
    ax.set_ylim(bottom=max(0, min_ppl - margin), top=max_ppl + margin * 2)

    # Add note about expected behavior
    ax.annotate('Note: All versions should have identical perplexity\n'
                '(KV-cache only affects generation, not forward pass)',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=8, ha='left', va='top', style='italic', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # Add system info
    first_data = next(iter(perplexity_data.values()))
    fig.text(0.5, 0.02,
             f"GPU: {first_data.get('gpu_name', 'N/A')} | "
             f"Dataset: {first_data.get('dataset', 'WikiText-2')} | "
             f"Stride: {first_data.get('stride', 'N/A')}",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_perplexity.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_perplexity_degradation(results, output_dir):
    """
    Plot perplexity degradation from baseline as a horizontal bar chart.

    Shows the percentage difference from v0 baseline for each version.
    Useful for quickly identifying if any optimization degrades quality.
    """
    # Collect perplexity data
    perplexity_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'perplexity':
            if data:
                perplexity_data[version] = data[-1]

    if not perplexity_data or 'v0' not in perplexity_data:
        print("  No perplexity data or no v0 baseline. Skipping degradation plot.")
        return

    baseline_ppl = perplexity_data['v0']['perplexity']

    # Sort versions (excluding v0)
    versions = sorted([v for v in perplexity_data.keys() if v != 'v0'])
    if not versions:
        print("  Only v0 found. Skipping degradation plot.")
        return

    degradations = []
    for v in versions:
        ppl = perplexity_data[v]['perplexity']
        deg = ((ppl - baseline_ppl) / baseline_ppl) * 100
        degradations.append(deg)

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(versions))
    colors = ['#2ecc71' if d <= 0.1 else '#e74c3c' if d > 1.0 else '#f39c12'
              for d in degradations]

    bars = ax.barh(y_pos, degradations, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, deg in zip(bars, degradations):
        width = bar.get_width()
        label_x = width + 0.05 if width >= 0 else width - 0.05
        ha = 'left' if width >= 0 else 'right'
        ax.annotate(f'{deg:+.3f}%',
                    xy=(label_x, bar.get_y() + bar.get_height() / 2),
                    va='center', ha=ha, fontsize=10, fontweight='bold')

    # Labels
    version_labels = [get_cache_label(v) for v in versions]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(version_labels)

    ax.set_xlabel('Perplexity Degradation from Baseline (%)', fontweight='bold')
    ax.set_ylabel('Version', fontweight='bold')
    ax.set_title('Perplexity Degradation Relative to v0 (No Cache)', fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    # Add threshold annotations
    ax.axvline(x=0.1, color='green', linewidth=1, linestyle='--', alpha=0.7)
    ax.axvline(x=1.0, color='red', linewidth=1, linestyle='--', alpha=0.7)

    # Legend for thresholds
    ax.annotate('Green: ≤0.1% | Yellow: 0.1-1% | Red: >1%',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom', style='italic', color='#555555')

    first_data = next(iter(perplexity_data.values()))
    fig.text(0.5, 0.02,
             f"Baseline (v0) perplexity: {baseline_ppl:.2f} | "
             f"Dataset: {first_data.get('dataset', 'WikiText-2')}",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_perplexity_degradation.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_max_batch_capacity(results, output_dir):
    """
    Plot maximum batch size capacity comparison across versions as a bar chart.

    This shows the maximum batch size each version can handle for inference
    without running out of memory.
    """
    # Collect capacity data
    capacity_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'max_batch_capacity':
            if data:
                # Take the latest record for each version
                capacity_data[version] = data[-1]

    if not capacity_data:
        print("  No max_batch_capacity data found. Skipping capacity plot.")
        return

    # Sort versions
    versions = sorted(capacity_data.keys())
    max_batches = [capacity_data[v]['max_batch_size'] for v in versions]
    peak_memory_gb = [capacity_data[v].get('peak_memory_gb', 0) for v in versions]

    # Get baseline (v0) for comparison
    baseline_batch = capacity_data.get('v0', {}).get('max_batch_size', max_batches[0])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    x_pos = np.arange(len(versions))
    colors = [VERSION_COLORS.get(v, '#666666') for v in versions]

    bars = ax.bar(x_pos, max_batches, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, batch, mem) in enumerate(zip(bars, max_batches, peak_memory_gb)):
        height = bar.get_height()
        # Calculate improvement from baseline
        if baseline_batch > 0 and versions[i] != 'v0':
            improvement = ((batch - baseline_batch) / baseline_batch) * 100
            imp_str = f"\n({improvement:+.1f}%)"
        elif versions[i] == 'v0':
            imp_str = "\n(baseline)"
        else:
            imp_str = ""
        ax.annotate(f'{batch}{imp_str}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Labels
    version_labels = [get_cache_label(v) for v in versions]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(version_labels, rotation=15, ha='right')

    ax.set_xlabel('Version', fontweight='bold')
    ax.set_ylabel('Maximum Batch Size', fontweight='bold')

    # Get capacity_T from data
    capacity_T = capacity_data[versions[0]].get('capacity_T', 128)
    ax.set_title(f'Maximum Inference Batch Size by Version (T={capacity_T})', fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0, top=max(max_batches) * 1.2)

    # Add annotation about what this measures
    ax.annotate('Higher is better\n(more parallel sequences)',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=9, ha='right', va='top', style='italic', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # Add system info
    first_data = next(iter(capacity_data.values()))
    fig.text(0.5, 0.02,
             f"GPU: {first_data.get('gpu_name', 'N/A')} | "
             f"Prompt Length: {first_data.get('prompt_length', 'N/A')} | "
             f"dtype: {first_data.get('dtype', 'N/A')}",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_max_batch_capacity.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()

    # --- Secondary plot: Memory at Max Batch ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    bars2 = ax2.bar(x_pos, peak_memory_gb, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, mem in zip(bars2, peak_memory_gb):
        height = bar.get_height()
        ax2.annotate(f'{mem:.2f} GB',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(version_labels, rotation=15, ha='right')

    ax2.set_xlabel('Version', fontweight='bold')
    ax2.set_ylabel('Peak Memory at Max Batch (GB)', fontweight='bold')
    ax2.set_title(f'Peak Memory Usage at Maximum Batch Size (T={capacity_T})', fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    fig2.text(0.5, 0.02,
              f"GPU: {first_data.get('gpu_name', 'N/A')} | "
              f"Shows memory used when running at max capacity",
              ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path2 = Path(output_dir) / 'comparison_peak_memory_at_max_batch.png'
    plt.savefig(path2, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path2}")
    plt.close()


def export_latex_tables(results, output_dir):
    """
    Export publication-ready LaTeX tables for benchmark results.
    Creates tables for:
    1. Latency comparison across versions
    2. VRAM comparison across versions
    3. Summary statistics table
    4. Perplexity comparison table
    """
    # Collect data
    latency_data = {}
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = {d['T']: d for d in data}
        elif benchmark_name == 'vram_vs_T':
            vram_data[version] = {d['T']: d for d in data}

    if not latency_data:
        print("  No latency data for LaTeX export. Skipping.")
        return

    # Get sorted versions and T values
    versions = sorted(latency_data.keys())
    all_T = set()
    for v_data in latency_data.values():
        all_T.update(v_data.keys())
    T_values = sorted(all_T)

    # Version labels for table headers
    version_headers = {
        'v0': r'No Cache',
        'v1': r'KV-Cache',
        'v2': r'KV+INT8',
        'v3': r'KV+Share',
        'v4': r'KV+INT8+Share',
    }

    # --- Table 1: Per-Token Latency Comparison ---
    latex_lines = []
    latex_lines.append(r"% Per-Token Latency Comparison Table")
    latex_lines.append(r"% Auto-generated by plot_results.py")
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \caption{Per-Token Generation Latency (ms) Across Cache Configurations}")
    latex_lines.append(r"  \label{tab:per_token_latency}")

    # Build column spec
    col_spec = "r" + "r" * len(versions)
    latex_lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"    \toprule")

    # Header row
    header = r"    $T$ & " + " & ".join([version_headers.get(v, v) for v in versions]) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"    \midrule")

    # Data rows
    for T in T_values:
        row_values = [str(T)]
        for v in versions:
            if v in latency_data and T in latency_data[v]:
                val = latency_data[v][T]['time_per_token_ms_median']
                row_values.append(f"{val:.2f}")
            else:
                row_values.append("--")
        latex_lines.append("    " + " & ".join(row_values) + r" \\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    # --- Table 2: Total Latency Comparison ---
    latex_lines.append(r"% Total Latency Comparison Table")
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \caption{Total Generation Latency (ms) Across Cache Configurations}")
    latex_lines.append(r"  \label{tab:total_latency}")
    latex_lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(header)
    latex_lines.append(r"    \midrule")

    for T in T_values:
        row_values = [str(T)]
        for v in versions:
            if v in latency_data and T in latency_data[v]:
                val = latency_data[v][T]['time_total_ms_median']
                row_values.append(f"{val:.1f}")
            else:
                row_values.append("--")
        latex_lines.append("    " + " & ".join(row_values) + r" \\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    # --- Table 3: VRAM Usage Comparison ---
    if vram_data:
        vram_versions = sorted(vram_data.keys())
        vram_T = set()
        for v_data in vram_data.values():
            vram_T.update(v_data.keys())
        vram_T_values = sorted(vram_T)

        vram_col_spec = "r" + "r" * len(vram_versions)
        vram_header = r"    $T$ & " + " & ".join([version_headers.get(v, v) for v in vram_versions]) + r" \\"

        latex_lines.append(r"% VRAM Usage Comparison Table")
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"  \centering")
        latex_lines.append(r"  \caption{Peak VRAM Usage (MB) Across Cache Configurations}")
        latex_lines.append(r"  \label{tab:vram_usage}")
        latex_lines.append(r"  \begin{tabular}{" + vram_col_spec + r"}")
        latex_lines.append(r"    \toprule")
        latex_lines.append(vram_header)
        latex_lines.append(r"    \midrule")

        for T in vram_T_values:
            row_values = [str(T)]
            for v in vram_versions:
                if v in vram_data and T in vram_data[v]:
                    val = vram_data[v][T]['peak_memory_bytes'] / 1e6
                    row_values.append(f"{val:.1f}")
                else:
                    row_values.append("--")
            latex_lines.append("    " + " & ".join(row_values) + r" \\")

        latex_lines.append(r"    \bottomrule")
        latex_lines.append(r"  \end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

    # --- Table 4: Summary Statistics ---
    latex_lines.append(r"% Summary Statistics Table")
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \caption{Performance Summary at $T=1024$ Tokens}")
    latex_lines.append(r"  \label{tab:summary}")
    latex_lines.append(r"  \begin{tabular}{lrrrr}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Configuration & Latency (ms) & Per-Token (ms) & Throughput (tok/s) & VRAM (MB) \\")
    latex_lines.append(r"    \midrule")

    T_summary = 1024
    for v in versions:
        label = version_headers.get(v, v)
        if v in latency_data and T_summary in latency_data[v]:
            lat = latency_data[v][T_summary]
            total_ms = lat['time_total_ms_median']
            per_tok_ms = lat['time_per_token_ms_median']
            throughput = 1000.0 / per_tok_ms

            if v in vram_data and T_summary in vram_data[v]:
                vram_mb = vram_data[v][T_summary]['peak_memory_bytes'] / 1e6
                vram_str = f"{vram_mb:.1f}"
            else:
                vram_str = "--"

            latex_lines.append(f"    {label} & {total_ms:.1f} & {per_tok_ms:.2f} & {throughput:.1f} & {vram_str} \\\\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    # --- Table 5: Speedup Table (relative to v0) ---
    if 'v0' in latency_data:
        latex_lines.append(r"% Speedup Table (relative to No Cache baseline)")
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"  \centering")
        latex_lines.append(r"  \caption{Speedup Factor Relative to No-Cache Baseline}")
        latex_lines.append(r"  \label{tab:speedup}")

        other_versions = [v for v in versions if v != 'v0']
        speedup_col_spec = "r" + "r" * len(other_versions)
        speedup_header = r"    $T$ & " + " & ".join([version_headers.get(v, v) for v in other_versions]) + r" \\"

        latex_lines.append(r"  \begin{tabular}{" + speedup_col_spec + r"}")
        latex_lines.append(r"    \toprule")
        latex_lines.append(speedup_header)
        latex_lines.append(r"    \midrule")

        for T in T_values:
            if T not in latency_data['v0']:
                continue
            v0_time = latency_data['v0'][T]['time_total_ms_median']
            row_values = [str(T)]
            for v in other_versions:
                if v in latency_data and T in latency_data[v]:
                    v_time = latency_data[v][T]['time_total_ms_median']
                    speedup = v0_time / v_time
                    row_values.append(f"{speedup:.2f}$\\times$")
                else:
                    row_values.append("--")
            latex_lines.append("    " + " & ".join(row_values) + r" \\")

        latex_lines.append(r"    \bottomrule")
        latex_lines.append(r"  \end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

    # --- Table 6: Scaling Analysis Table ---
    # Shows O(T²) coefficient, O(T) coefficient, R², and scaling class
    latex_lines.append(r"% Scaling Analysis Table")
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \caption{Latency Scaling Analysis (Quadratic Fit: $aT^2 + bT + c$)}")
    latex_lines.append(r"  \label{tab:scaling}")
    latex_lines.append(r"  \begin{tabular}{lrrrl}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Configuration & $a$ ($\times 10^{-3}$) & $b$ & $R^2$ & Scaling \\")
    latex_lines.append(r"    \midrule")

    for v in versions:
        label = version_headers.get(v, v)
        v_data = sorted(latency_data[v].values(), key=lambda x: x['T'])
        if len(v_data) < 3:
            continue  # Need at least 3 points for quadratic fit

        T_vals = [d['T'] for d in v_data]
        total_ms = [d['time_total_ms_median'] for d in v_data]

        a, b, c = quadratic_fit(T_vals, total_ms)
        T_dense = np.asarray(T_vals, dtype=float)
        y_pred = a * T_dense**2 + b * T_dense + c
        r2 = r2_score(total_ms, y_pred)

        # Classify scaling behavior
        if abs(a) < 1e-5:
            scaling_class = r"$\approx O(T)$"
        elif a > 0:
            scaling_class = r"$O(T^2)$"
        else:
            scaling_class = r"sub-linear"

        latex_lines.append(f"    {label} & {a*1000:.3f} & {b:.2f} & {r2:.3f} & {scaling_class} \\\\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    # --- Table 7: Statistical Summary Table ---
    # Shows Mean ± CI, Min, Max, CV% for each version
    latex_lines.append(r"% Statistical Summary Table")
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \caption{Per-Token Latency Statistics Across All $T$ Values}")
    latex_lines.append(r"  \label{tab:stats}")
    latex_lines.append(r"  \begin{tabular}{lrrrr}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Configuration & Mean (ms) & Min (ms) & Max (ms) & CV (\%) \\")
    latex_lines.append(r"    \midrule")

    for v in versions:
        label = version_headers.get(v, v)
        v_data = list(latency_data[v].values())
        if not v_data:
            continue

        per_tok_vals = [d['time_per_token_ms_median'] for d in v_data]
        mean_val = np.mean(per_tok_vals)
        min_val = np.min(per_tok_vals)
        max_val = np.max(per_tok_vals)
        std_val = np.std(per_tok_vals)
        cv_pct = (std_val / mean_val * 100) if mean_val > 0 else 0

        latex_lines.append(f"    {label} & {mean_val:.2f} & {min_val:.2f} & {max_val:.2f} & {cv_pct:.1f} \\\\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    # --- Table 8: Perplexity Comparison Table ---
    perplexity_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'perplexity':
            if data:
                perplexity_data[version] = data[-1]

    if perplexity_data:
        ppl_versions = sorted(perplexity_data.keys())
        baseline_ppl = perplexity_data.get('v0', {}).get('perplexity', None)

        latex_lines.append(r"% Perplexity Comparison Table")
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"  \centering")
        latex_lines.append(r"  \caption{Perplexity on WikiText-2 (Lower is Better)}")
        latex_lines.append(r"  \label{tab:perplexity}")
        latex_lines.append(r"  \begin{tabular}{lrrr}")
        latex_lines.append(r"    \toprule")
        latex_lines.append(r"    Configuration & Perplexity & Degradation (\%) & Tokens Evaluated \\")
        latex_lines.append(r"    \midrule")

        for v in ppl_versions:
            label = version_headers.get(v, v)
            ppl = perplexity_data[v]['perplexity']
            n_tokens = perplexity_data[v].get('num_tokens_evaluated', '--')

            if baseline_ppl and v != 'v0':
                deg = ((ppl - baseline_ppl) / baseline_ppl) * 100
                deg_str = f"{deg:+.3f}"
            elif v == 'v0':
                deg_str = "baseline"
            else:
                deg_str = "--"

            n_tok_str = f"{n_tokens:,}" if isinstance(n_tokens, int) else str(n_tokens)
            latex_lines.append(f"    {label} & {ppl:.2f} & {deg_str} & {n_tok_str} \\\\")

        latex_lines.append(r"    \bottomrule")
        latex_lines.append(r"  \end{tabular}")
        latex_lines.append(r"\end{table}")

    # Write to file
    latex_content = "\n".join(latex_lines)
    out_path = Path(output_dir) / 'benchmark_tables.tex'
    with open(out_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: {out_path}")


def export_perplexity_markdown_table(results, output_dir):
    """
    Export perplexity results as a markdown table.
    Includes all versions (v0-v4, v3a, v4a, etc.) with comparison to v1 baseline.
    """
    from collections import defaultdict

    # Collect perplexity data
    perplexity_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'perplexity' and data:
            perplexity_data[version] = data[-1]  # Latest result

    if not perplexity_data:
        print("  No perplexity data found. Skipping markdown table.")
        return

    # Get baseline
    baseline_ppl = perplexity_data.get('v1', {}).get('perplexity')

    # Build markdown table
    lines = [
        "# Perplexity Results",
        "",
        f"Generated from benchmark results. Baseline: v1 = {baseline_ppl:.2f}" if baseline_ppl else "",
        "",
        "| Version | Description | Perplexity | vs v1 | Tokens |",
        "|---------|-------------|------------|-------|--------|",
    ]

    for version in sorted(perplexity_data.keys()):
        data = perplexity_data[version]
        ppl = data['perplexity']
        desc = data.get('version_description', version)
        tokens = data.get('num_tokens_evaluated', '-')

        if version == 'v1':
            vs_v1 = "baseline"
        elif baseline_ppl:
            ratio = ppl / baseline_ppl
            if ratio > 10:
                vs_v1 = f"**{ratio:.1f}x worse**"
            elif ratio > 1.5:
                vs_v1 = f"{ratio:.1f}x worse"
            else:
                vs_v1 = f"{ratio:.2f}x"
        else:
            vs_v1 = "-"

        # Highlight experimental versions
        if version in ['v3a', 'v4a']:
            lines.append(f"| **{version}** | **{desc}** | **{ppl:.2f}** | {vs_v1} | {tokens} |")
        elif ppl > 500:
            lines.append(f"| {version} | {desc} | **{ppl:.2f}** | {vs_v1} | {tokens} |")
        else:
            lines.append(f"| {version} | {desc} | {ppl:.2f} | {vs_v1} | {tokens} |")

    lines.extend([
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "- **v0-v2**: Normal perplexity (~30-35)",
        "- **v3, v4**: Cross-layer sharing breaks quality (~800 PPL)",
        "- **v3a, v4a**: Q-alignment improves to ~220 PPL (3.4-3.8x better than v3/v4)",
        "",
        "See `CROSS_LAYER_RESEARCH.md` for detailed analysis.",
    ])

    # Write file
    out_path = Path(output_dir) / 'perplexity_table.md'
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# =============================================================================
# Split Comparison Plots: v0 vs v1, and KV-cache variants (v1 vs v2/v3/v4)
# =============================================================================

def plot_v0_vs_v1_latency(results, output_dir):
    """
    Focused comparison: v0 (no cache) vs v1 (KV-cache).
    Shows the fundamental O(T²) vs O(T) scaling difference.
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T' and version in ('v0', 'v1'):
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if 'v0' not in latency_data or 'v1' not in latency_data:
        print("  Need both v0 and v1 for v0 vs v1 comparison. Skipping.")
        return

    first_data = latency_data['v0'][0]

    # --- Per-token latency: v0 vs v1 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for version in ['v0', 'v1']:
        data = latency_data[version]
        T_values = [d['T'] for d in data]
        time_per_token = [d['time_per_token_ms_median'] for d in data]
        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(T_values, time_per_token, f'{marker}-',
                markersize=7, linewidth=2.5,
                color=color, label=label)

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax.set_title('KV-Cache Impact: O(T) → O(1) Per-Token Scaling', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add scaling annotations
    v0_T = [d['T'] for d in latency_data['v0']]
    v0_pt = [d['time_per_token_ms_median'] for d in latency_data['v0']]
    v1_pt = [d['time_per_token_ms_median'] for d in latency_data['v1']]

    # Annotate at max T
    max_idx = -1
    ax.annotate(f'v0: {v0_pt[max_idx]:.1f} ms/tok',
                xy=(v0_T[max_idx], v0_pt[max_idx]),
                xytext=(v0_T[max_idx] - 150, v0_pt[max_idx] + 0.3),
                fontsize=10, color=VERSION_COLORS['v0'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=VERSION_COLORS['v0']))
    ax.annotate(f'v1: {v1_pt[max_idx]:.1f} ms/tok',
                xy=(v0_T[max_idx], v1_pt[max_idx]),
                xytext=(v0_T[max_idx] - 150, v1_pt[max_idx] - 0.5),
                fontsize=10, color=VERSION_COLORS['v1'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=VERSION_COLORS['v1']))

    # Add speedup annotation
    speedup = v0_pt[max_idx] / v1_pt[max_idx]
    ax.annotate(f'{speedup:.1f}x faster at T={v0_T[max_idx]}',
                xy=(0.98, 0.5), xycoords='axes fraction',
                fontsize=12, ha='right', va='center', fontweight='bold',
                color=VERSION_COLORS['v1'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | v0: no cache (recompute all) | v1: KV-cache (incremental)",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_v0_vs_v1_latency.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_kv_variants_latency(results, output_dir):
    """
    Compare KV-cache variants: v1 vs v2 vs v3 vs v4.
    Excludes v0 to zoom in on the differences between cache implementations.
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T' and version in ('v1', 'v2', 'v3', 'v4'):
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if len(latency_data) < 2:
        print("  Need at least 2 KV-cache variants for comparison. Skipping.")
        return

    first_data = next(iter(latency_data.values()))[0]

    # --- Per-token latency comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for version in sorted(latency_data.keys()):
        data = latency_data[version]
        T_values = [d['T'] for d in data]
        time_per_token = [d['time_per_token_ms_median'] for d in data]
        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(T_values, time_per_token, f'{marker}-',
                markersize=6, linewidth=2,
                color=color, label=label)

    # Add horizontal reference line at v1 baseline
    if 'v1' in latency_data:
        v1_avg = np.mean([d['time_per_token_ms_median'] for d in latency_data['v1']])
        ax.axhline(y=v1_avg, color=VERSION_COLORS['v1'], linestyle='--', linewidth=1.5, alpha=0.5)
        ax.annotate(f'v1 avg: {v1_avg:.2f} ms', xy=(0.02, v1_avg),
                    xycoords=('axes fraction', 'data'),
                    fontsize=9, color=VERSION_COLORS['v1'], va='bottom')

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Time per Token (ms)', fontweight='bold')
    ax.set_title('KV-Cache Variants: Per-Token Latency Comparison', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Comparing KV-cache optimizations (v0 excluded for clarity)",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_kv_variants_latency.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_speedup_vs_v1(results, output_dir):
    """
    Plot relative speedup of KV-cache variants compared to v1 baseline.
    Speedup > 1.0 means faster than v1, < 1.0 means slower.
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = {d['T']: d['time_per_token_ms_median'] for d in data}

    if 'v1' not in latency_data:
        print("  Need v1 baseline for speedup comparison. Skipping.")
        return

    v1_data = latency_data['v1']
    variants = [v for v in latency_data.keys() if v != 'v1' and v != 'v0']

    if not variants:
        print("  No KV-cache variants (v2, v3, v4) found. Skipping speedup plot.")
        return

    first_data_key = next(iter(results.keys()))
    first_data = results[first_data_key][0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for version in sorted(variants):
        v_data = latency_data[version]
        common_T = sorted(set(v1_data.keys()) & set(v_data.keys()))
        if not common_T:
            continue

        # Speedup = v1_time / variant_time (>1 means variant is faster)
        speedup = [v1_data[T] / v_data[T] for T in common_T]
        label = get_cache_label(version)
        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.plot(common_T, speedup, f'{marker}-',
                markersize=6, linewidth=2,
                color=color, label=label)

    # Reference line at 1.0 (v1 baseline)
    ax.axhline(y=1.0, color=VERSION_COLORS['v1'], linestyle='-', linewidth=2,
               label='v1 baseline (1.0x)')

    # Shade regions
    T_range = ax.get_xlim()
    ax.fill_between([T_range[0], T_range[1]], 1.0, 2.0,
                    alpha=0.1, color='green', label='Faster than v1')
    ax.fill_between([T_range[0], T_range[1]], 0, 1.0,
                    alpha=0.1, color='red', label='Slower than v1')

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold')
    ax.set_ylabel('Speedup vs v1 (higher = faster)', fontweight='bold')
    ax.set_title('KV-Cache Variants: Speedup Relative to v1 Baseline', fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.text(0.5, 0.02,
             f"GPU: {first_data['gpu_name']} | Speedup = v1_latency / variant_latency",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'comparison_speedup_vs_v1.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


# =============================================================================
# NEW SPLIT COMPARISON PLOTS - Avoid v0 distortion
# =============================================================================

def add_standard_footer(fig, gpu="A6000", dtype="fp16", extra=""):
    """Add standardized footer to plots - consistent metadata display."""
    footer = f"GPU: {gpu} | {dtype}"
    if extra:
        footer += f" | {extra}"
    fig.text(0.5, 0.01, footer, ha='center', fontsize=8, style='italic', color='#666666')


def plot_v0_v1_total_latency_with_projection(results, output_dir, T_max=4096):
    """
    v0 vs v1 total latency with projection to T_max.
    Shows quadratic (v0) vs linear (v1) scaling with R² values.
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T' and version in ('v0', 'v1'):
            latency_data[version] = sorted(data, key=lambda x: x['T'])

    if 'v0' not in latency_data or 'v1' not in latency_data:
        print("  Need both v0 and v1 for projection plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    # Extract data
    v0_T = np.array([d['T'] for d in latency_data['v0']])
    v0_time = np.array([d['time_total_ms_median'] for d in latency_data['v0']])
    v0_std = np.array([d['time_total_ms_std'] for d in latency_data['v0']])

    v1_T = np.array([d['T'] for d in latency_data['v1']])
    v1_time = np.array([d['time_total_ms_median'] for d in latency_data['v1']])
    v1_std = np.array([d['time_total_ms_std'] for d in latency_data['v1']])

    max_measured_T = max(v0_T.max(), v1_T.max())

    # Fit models
    a0, b0, c0 = quadratic_fit(v0_T, v0_time)
    v0_fit_vals = a0 * v0_T**2 + b0 * v0_T + c0
    r2_v0 = r2_score(v0_time, v0_fit_vals)

    # For v1, fit linear model (should be nearly linear with cache)
    m1, b1 = linear_fit(v1_T, v1_time)
    v1_fit_vals = m1 * v1_T + b1
    r2_v1 = r2_score(v1_time, v1_fit_vals)

    # Generate projection range
    T_measured = np.linspace(min(v0_T.min(), v1_T.min()), max_measured_T, 200)
    T_projected = np.linspace(max_measured_T, T_max, 200)

    v0_measured_fit = a0 * T_measured**2 + b0 * T_measured + c0
    v0_projected_fit = a0 * T_projected**2 + b0 * T_projected + c0

    v1_measured_fit = m1 * T_measured + b1
    v1_projected_fit = m1 * T_projected + b1

    # Plot measured data points with error bars
    ax.errorbar(v0_T, v0_time, yerr=v0_std, fmt='o', markersize=6,
                color=VERSION_COLORS['v0'], ecolor=VERSION_COLORS['v0'],
                capsize=3, alpha=0.8, label='v0 measured', zorder=5)
    ax.errorbar(v1_T, v1_time, yerr=v1_std, fmt='s', markersize=6,
                color=VERSION_COLORS['v1'], ecolor=VERSION_COLORS['v1'],
                capsize=3, alpha=0.8, label='v1 measured', zorder=5)

    # Plot fit lines (solid for measured, dashed for projected)
    ax.plot(T_measured, v0_measured_fit, '-', color=VERSION_COLORS['v0'],
            linewidth=2.5, alpha=0.9)
    ax.plot(T_projected, v0_projected_fit, '--', color=VERSION_COLORS['v0'],
            linewidth=2.5, alpha=0.7, label='v0 projected')

    ax.plot(T_measured, v1_measured_fit, '-', color=VERSION_COLORS['v1'],
            linewidth=2.5, alpha=0.9)
    ax.plot(T_projected, v1_projected_fit, '--', color=VERSION_COLORS['v1'],
            linewidth=2.5, alpha=0.7, label='v1 projected')

    # Shade projection zone
    ax.axvspan(max_measured_T, T_max, alpha=0.08, color='gray', label='Projection zone')
    ax.axvline(x=max_measured_T, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # Calculate projected values at T_max
    v0_at_Tmax = a0 * T_max**2 + b0 * T_max + c0
    v1_at_Tmax = m1 * T_max + b1
    speedup_at_Tmax = v0_at_Tmax / v1_at_Tmax

    # Add R² and equation annotation box
    textstr = (f"v0: {a0:.5f}·T² + {b0:.2f}·T + {c0:.1f}\n"
               f"     R² = {r2_v0:.6f} (quadratic)\n\n"
               f"v1: {m1:.2f}·T + {b1:.1f}\n"
               f"     R² = {r2_v1:.4f} (linear)")
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9))

    # Add projection insight
    insight = (f"At T={T_max}:\n"
               f"v0: {v0_at_Tmax/1000:.1f}s\n"
               f"v1: {v1_at_Tmax/1000:.1f}s\n"
               f"Speedup: {speedup_at_Tmax:.1f}x")
    ax.text(0.97, 0.65, insight, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', ha='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9))

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Generation Time (ms)', fontweight='bold', fontsize=12)
    ax.set_title(f'Total Latency: v0 vs v1 with Projection to T={T_max}',
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(left=0, right=T_max * 1.02)
    ax.set_ylim(bottom=0)

    # Extract GPU name (short form)
    gpu_name = latency_data['v0'][0].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    dtype = latency_data['v0'][0].get('dtype', 'fp16')
    add_standard_footer(fig, gpu=gpu_short, dtype=dtype, extra=f"Projection beyond T={max_measured_T}")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / 'v0_v1_total_latency_projection.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_kv_variants_per_token_bar(results, output_dir, T_target=1024):
    """
    Bar chart comparing v1-v4 per-token latency at fixed T.
    Better than line chart since values are nearly constant.
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T' and version in ('v1', 'v2', 'v3', 'v4'):
            by_T = {d['T']: d for d in data}
            if T_target in by_T:
                latency_data[version] = by_T[T_target]

    if len(latency_data) < 2:
        print(f"  Need at least 2 KV variants with data at T={T_target}. Skipping.")
        return

    versions = sorted(latency_data.keys())
    latencies = [latency_data[v]['time_per_token_ms_median'] for v in versions]
    stds = [latency_data[v].get('time_per_token_ms_std', 0) for v in versions]
    colors = [VERSION_COLORS.get(v, '#666666') for v in versions]

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(versions))
    bars = ax.bar(x, latencies, yerr=stds, capsize=6,
                  color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val, std in zip(bars, latencies, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Reference line for v1 baseline
    v1_latency = latency_data.get('v1', {}).get('time_per_token_ms_median', latencies[0])
    ax.axhline(y=v1_latency, color=VERSION_COLORS['v1'], linestyle='--',
               linewidth=2, alpha=0.7, label=f'v1 baseline ({v1_latency:.2f} ms)')

    ax.set_ylabel('Per-Token Latency (ms)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Version', fontweight='bold', fontsize=12)
    ax.set_title(f'KV-Cache Variants: Per-Token Latency at T={T_target}',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions], fontsize=10)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add insight annotation
    best_v = versions[np.argmin(latencies)]
    worst_v = versions[np.argmax(latencies)]
    ax.annotate(f'Best: {best_v} ({min(latencies):.2f} ms)\nWorst: {worst_v} ({max(latencies):.2f} ms)',
                xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    gpu_name = list(latency_data.values())[0].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    add_standard_footer(fig, gpu=gpu_short, dtype='fp16', extra=f'T={T_target}, Batch=1')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / f'kv_variants_per_token_bar_T{T_target}.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_kv_variants_throughput_bar(results, output_dir, T_target=1024):
    """
    Bar chart comparing v1-v4 throughput (tokens/sec) at fixed T.
    """
    latency_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T' and version in ('v1', 'v2', 'v3', 'v4'):
            by_T = {d['T']: d for d in data}
            if T_target in by_T:
                latency_data[version] = by_T[T_target]

    if len(latency_data) < 2:
        print(f"  Need at least 2 KV variants with data at T={T_target}. Skipping.")
        return

    versions = sorted(latency_data.keys())
    throughputs = [1000.0 / latency_data[v]['time_per_token_ms_median'] for v in versions]
    # Error propagation: delta_tput = tput * (delta_t / t)
    stds = []
    for v in versions:
        t = latency_data[v]['time_per_token_ms_median']
        t_std = latency_data[v].get('time_per_token_ms_std', 0)
        tput = 1000.0 / t
        stds.append(tput * (t_std / t) if t > 0 else 0)

    colors = [VERSION_COLORS.get(v, '#666666') for v in versions]

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(versions))
    bars = ax.bar(x, throughputs, yerr=stds, capsize=6,
                  color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val, std in zip(bars, throughputs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Throughput (tokens/sec)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Version', fontweight='bold', fontsize=12)
    ax.set_title(f'KV-Cache Variants: Throughput at T={T_target}',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions], fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=0)

    gpu_name = list(latency_data.values())[0].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    add_standard_footer(fig, gpu=gpu_short, dtype='fp16', extra=f'T={T_target}, Batch=1')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / f'kv_variants_throughput_bar_T{T_target}.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_kv_variants_vram_bar(results, output_dir, T_target=1024):
    """
    Bar chart comparing v1-v4 VRAM usage at fixed T.
    """
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'vram_vs_T' and version in ('v1', 'v2', 'v3', 'v4'):
            by_T = {d['T']: d for d in data}
            if T_target in by_T:
                vram_data[version] = by_T[T_target]

    if len(vram_data) < 2:
        print(f"  Need at least 2 KV variants with VRAM data at T={T_target}. Skipping.")
        return

    versions = sorted(vram_data.keys())
    vram_mb = [vram_data[v]['peak_memory_bytes'] / 1e6 for v in versions]
    colors = [VERSION_COLORS.get(v, '#666666') for v in versions]

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(versions))
    bars = ax.bar(x, vram_mb, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, vram_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Reference line for v1 baseline
    v1_vram = vram_data.get('v1', {}).get('peak_memory_bytes', 0) / 1e6
    if v1_vram > 0:
        ax.axhline(y=v1_vram, color=VERSION_COLORS['v1'], linestyle='--',
                   linewidth=2, alpha=0.7, label=f'v1 baseline ({v1_vram:.0f} MB)')

    ax.set_ylabel('Peak VRAM (MB)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Version', fontweight='bold', fontsize=12)
    ax.set_title(f'KV-Cache Variants: VRAM Usage at T={T_target}',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions], fontsize=10)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Set y-axis to start from reasonable value
    min_vram = min(vram_mb) * 0.95
    ax.set_ylim(bottom=min_vram)

    # Add memory savings annotation
    if 'v1' in vram_data and 'v4' in vram_data:
        v4_vram = vram_data['v4']['peak_memory_bytes'] / 1e6
        savings = (1 - v4_vram / v1_vram) * 100 if v1_vram > 0 else 0
        ax.annotate(f'v4 saves {savings:.0f}% vs v1',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9))

    gpu_name = list(vram_data.values())[0].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    add_standard_footer(fig, gpu=gpu_short, dtype='fp16', extra=f'T={T_target}')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / f'kv_variants_vram_bar_T{T_target}.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_memory_latency_tradeoff_improved(results, output_dir, T_target=1024):
    """
    Improved tradeoff plot at T=1024 where v0's penalty is clear.
    v0 shown with hollow marker to distinguish from KV variants.
    """
    latency_data = {}
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'latency_vs_T':
            latency_data[version] = {d['T']: d for d in data}
        elif benchmark_name == 'vram_vs_T':
            vram_data[version] = {d['T']: d for d in data}

    versions_with_data = []
    for v in set(latency_data.keys()) & set(vram_data.keys()):
        if T_target in latency_data[v] and T_target in vram_data[v]:
            versions_with_data.append(v)

    if len(versions_with_data) < 2:
        print(f"  Need at least 2 versions at T={T_target} for tradeoff plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate v0 from KV variants
    kv_versions = [v for v in versions_with_data if v != 'v0']

    # Plot KV variants (filled markers)
    for version in sorted(kv_versions):
        vram_mb = vram_data[version][T_target]['peak_memory_bytes'] / 1e6
        latency_ms = latency_data[version][T_target]['time_total_ms_median']

        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')

        ax.scatter(vram_mb, latency_ms, s=200, c=color, marker=marker,
                   edgecolors='white', linewidth=2, zorder=5,
                   label=VERSION_LABELS.get(version, version))

        ax.annotate(version, (vram_mb, latency_ms),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=color)

    # Plot v0 with hollow marker (different visual treatment)
    if 'v0' in versions_with_data:
        v0_vram = vram_data['v0'][T_target]['peak_memory_bytes'] / 1e6
        v0_latency = latency_data['v0'][T_target]['time_total_ms_median']

        ax.scatter(v0_vram, v0_latency, s=250, facecolors='none',
                   edgecolors=VERSION_COLORS['v0'], linewidths=3, marker='o',
                   zorder=5, label=f"v0: No cache (O(T²))")

        ax.annotate('v0', (v0_vram, v0_latency),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=VERSION_COLORS['v0'])

        # Add arrow showing v0's latency penalty
        if 'v1' in versions_with_data:
            v1_vram = vram_data['v1'][T_target]['peak_memory_bytes'] / 1e6
            v1_latency = latency_data['v1'][T_target]['time_total_ms_median']
            penalty = v0_latency / v1_latency

            # Draw arrow from v1 to v0
            ax.annotate('', xy=(v0_vram, v0_latency),
                        xytext=(v1_vram, v1_latency),
                        arrowprops=dict(arrowstyle='->', color='red',
                                       lw=2, connectionstyle='arc3,rad=0.2'))
            # Add penalty label
            mid_x = (v0_vram + v1_vram) / 2
            mid_y = (v0_latency + v1_latency) / 2
            ax.annotate(f'{penalty:.1f}x slower\n(no cache penalty)',
                        xy=(mid_x, mid_y), fontsize=10, color='red',
                        fontweight='bold', ha='center')

    # Shade "good" region (lower-left)
    ax.annotate('← Better (lower latency, lower VRAM)',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, ha='left', va='bottom', style='italic', color='#555555')

    ax.set_xlabel('Peak VRAM Usage (MB)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Total Latency at T={T_target} (ms)', fontweight='bold', fontsize=12)
    ax.set_title(f'Memory-Latency Tradeoff at T={T_target}', fontweight='bold', fontsize=13, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

    gpu_name = list(latency_data.values())[0][T_target].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    add_standard_footer(fig, gpu=gpu_short, dtype='fp16', extra=f'T={T_target}')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / f'tradeoff_memory_latency_T{T_target}.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_vram_growth_comparison(results, output_dir):
    """
    Compare VRAM growth between v0 and v1.
    Shows how memory scales differently with sequence length.
    """
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'vram_vs_T' and version in ('v0', 'v1'):
            vram_data[version] = sorted(data, key=lambda x: x['T'])

    if 'v0' not in vram_data or 'v1' not in vram_data:
        print("  Need both v0 and v1 for VRAM growth comparison. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for version in ['v0', 'v1']:
        data = vram_data[version]
        T_values = [d['T'] for d in data]
        peak_mb = [d['peak_memory_bytes'] / 1e6 for d in data]
        baseline = peak_mb[0]
        delta_mb = [m - baseline for m in peak_mb]

        color = VERSION_COLORS.get(version, '#666666')
        marker = VERSION_MARKERS.get(version, 'o')
        label = VERSION_LABELS.get(version, version)

        ax.plot(T_values, delta_mb, f'{marker}-', color=color, markersize=6,
                linewidth=2.5, label=label)

        # Add linear fit
        m, b = linear_fit(T_values, delta_mb)
        T_dense = np.linspace(min(T_values), max(T_values), 100)
        ax.plot(T_dense, m * T_dense + b, '--', color=color, alpha=0.5, linewidth=1.5)

    # Add insight box
    v0_data = vram_data['v0']
    v1_data = vram_data['v1']
    v0_delta = v0_data[-1]['peak_memory_bytes'] / 1e6 - v0_data[0]['peak_memory_bytes'] / 1e6
    v1_delta = v1_data[-1]['peak_memory_bytes'] / 1e6 - v1_data[0]['peak_memory_bytes'] / 1e6
    max_T = v0_data[-1]['T']

    insight = (f"Memory Growth (T=32 → T={max_T}):\n"
               f"v0: +{v0_delta:.1f} MB\n"
               f"v1: +{v1_delta:.1f} MB\n\n"
               f"v1 uses {v1_delta/v0_delta:.1f}x more growth\n"
               f"(KV-cache storage overhead)")
    ax.text(0.03, 0.97, insight, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    ax.set_xlabel('Number of Generated Tokens (T)', fontweight='bold', fontsize=12)
    ax.set_ylabel('VRAM Growth from Baseline (MB)', fontweight='bold', fontsize=12)
    ax.set_title('VRAM Growth: v0 vs v1', fontweight='bold', fontsize=13, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=0)

    gpu_name = vram_data['v0'][0].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    add_standard_footer(fig, gpu=gpu_short, dtype='fp16', extra=f'Baseline: T=32')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / 'vram_growth_v0_v1.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_per_phase_pie_with_percentages(results, output_dir):
    """
    Improved pie charts with percentages directly ON the pie slices.
    """
    phase_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'per_phase_timing' and version in ('v0', 'v1'):
            if data:
                phase_data[version] = data[-1]

    if 'v0' not in phase_data or 'v1' not in phase_data:
        print("  Need both v0 and v1 per-phase data for improved pie charts. Skipping.")
        return

    phase_order = ['embedding', 'attention', 'mlp', 'head', 'other']
    phase_labels = {
        'embedding': 'Embed', 'attention': 'Attn',
        'mlp': 'MLP', 'head': 'Head', 'other': 'Other'
    }
    phase_colors = {
        'embedding': '#06A77D', 'attention': '#2E86AB',
        'mlp': '#A23B72', 'head': '#F18F01', 'other': '#999999'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, version in enumerate(['v0', 'v1']):
        ax = axes[idx]
        times = phase_data[version]['time_phase_ms']
        total = times.get('total', sum(times.get(p, 0) for p in phase_order))

        values = []
        labels = []
        colors = []
        for p in phase_order:
            v = times.get(p, 0)
            if v > 0:
                values.append(v)
                labels.append(phase_labels[p])
                colors.append(phase_colors[p])

        # Use autopct to show percentages ON the pie
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
            textprops={'fontsize': 10},
            pctdistance=0.75
        )

        # Style the percentage text
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            autotext.set_color('white')

        cache_label = get_cache_label(version)
        T_star = phase_data[version].get('T_star', '?')
        ax.set_title(f'{cache_label}\nTotal: {total:.0f}ms (T={T_star})',
                     fontweight='bold', fontsize=12)

    fig.suptitle('Per-Phase Time Breakdown: v0 vs v1', fontweight='bold', fontsize=14, y=1.02)

    # Add key insight
    v0_attn = phase_data['v0']['time_phase_ms'].get('attention', 0)
    v1_attn = phase_data['v1']['time_phase_ms'].get('attention', 0)
    v0_total = phase_data['v0']['time_phase_ms'].get('total', 1)
    v1_total = phase_data['v1']['time_phase_ms'].get('total', 1)

    insight = (f"Key Finding: Attention is {v0_attn/v0_total*100:.0f}% (v0) vs {v1_attn/v1_total*100:.0f}% (v1)\n"
               f"Total time reduced by {(1-v1_total/v0_total)*100:.0f}%")
    fig.text(0.5, 0.02, insight, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    path = Path(output_dir) / 'per_phase_pie_improved.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_per_phase_breakdown_with_insights(results, output_dir):
    """
    Improved per-phase breakdown bar chart with key insights highlighted.
    """
    phase_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'per_phase_timing':
            for record in data:
                v = record.get('version', 'unknown')
                phase_data[v] = record

    if len(phase_data) < 2:
        print("  Need at least 2 versions for per-phase breakdown. Skipping.")
        return

    versions = sorted(phase_data.keys())
    T_star = phase_data[versions[0]]['T_star']

    phase_order = ['embedding', 'attention', 'mlp', 'head', 'other']
    phase_labels = {
        'embedding': 'Embedding', 'attention': 'Self-Attention',
        'mlp': 'MLP', 'head': 'LM Head', 'other': 'Other'
    }

    all_phases = set()
    for v in versions:
        all_phases.update(phase_data[v]['time_phase_ms'].keys())
    phases = [p for p in phase_order if p in all_phases and p != 'total']

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(phases))
    width = 0.8 / len(versions)

    for i, version in enumerate(versions):
        phase_times = phase_data[version]['time_phase_ms']
        times = [phase_times.get(p, 0) for p in phases]
        offset = (i - len(versions)/2 + 0.5) * width
        color = VERSION_COLORS.get(version, plt.cm.tab10(i / len(versions)))

        bars = ax.bar(x + offset, times, width, label=VERSION_LABELS.get(version, version),
                     color=color, edgecolor='white', linewidth=0.5)

        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{t:.0f}', ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Phase', fontweight='bold', fontsize=12)
    ax.set_title(f'Per-Phase Timing Breakdown by Version (T={T_star})', fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([phase_labels[p] for p in phases], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add key insights box
    if 'v0' in phase_data and 'v1' in phase_data:
        v0_times = phase_data['v0']['time_phase_ms']
        v1_times = phase_data['v1']['time_phase_ms']
        v0_total = v0_times.get('total', sum(v0_times.get(p, 0) for p in phases))
        v1_total = v1_times.get('total', sum(v1_times.get(p, 0) for p in phases))

        # Find biggest change
        changes = {}
        for p in phases:
            v0_p = v0_times.get(p, 0)
            v1_p = v1_times.get(p, 0)
            if v0_p > 0:
                changes[p] = ((v1_p - v0_p) / v0_p) * 100

        biggest_change = min(changes.items(), key=lambda x: x[1]) if changes else ('N/A', 0)

        insights = (f"Key Findings (T={T_star}):\n"
                   f"• Total: {v0_total:.0f}ms → {v1_total:.0f}ms ({(1-v1_total/v0_total)*100:+.0f}%)\n"
                   f"• Biggest reduction: {phase_labels.get(biggest_change[0], biggest_change[0])} ({biggest_change[1]:.0f}%)\n"
                   f"• KV-cache avoids redundant attention computation")
        ax.text(0.02, 0.98, insights, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    gpu_name = phase_data[versions[0]].get('gpu_name', 'GPU')
    gpu_short = gpu_name.split()[-1] if 'RTX' in gpu_name or 'A' in gpu_name else gpu_name
    add_standard_footer(fig, gpu=gpu_short, dtype='fp16', extra=f'T={T_star}')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / 'per_phase_breakdown_with_insights.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


# =============================================================================
# ORIGINAL FUNCTIONS CONTINUE BELOW
# =============================================================================

def plot_vram_bar_chart(results, output_dir):
    """
    Bar chart showing peak VRAM usage for each version.
    More meaningful than delta plots for paper figures.
    Shows baseline (T=32) and peak (max T) for each version.
    """
    vram_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'vram_vs_T':
            sorted_data = sorted(data, key=lambda x: x['T'])
            vram_data[version] = sorted_data

    if not vram_data:
        print("  No VRAM data found. Skipping VRAM bar chart.")
        return

    versions = sorted(vram_data.keys())
    first_data = next(iter(vram_data.values()))[0]

    # Extract baseline and peak for each version
    baseline_mb = []
    peak_mb = []
    for v in versions:
        data = vram_data[v]
        baseline_mb.append(data[0]['peak_memory_bytes'] / 1e6)
        peak_mb.append(data[-1]['peak_memory_bytes'] / 1e6)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(versions))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_mb, width, label=f'Baseline (T={vram_data[versions[0]][0]["T"]})',
                   color='#2A9D8F', edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, peak_mb, width, label=f'Peak (T={vram_data[versions[0]][-1]["T"]})',
                   color='#E63946', edgecolor='white', linewidth=1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('VRAM Usage (MB)', fontweight='bold')
    ax.set_xlabel('Version', fontweight='bold')
    ax.set_title('Peak VRAM Usage by Version', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([get_cache_label(v) for v in versions], rotation=15, ha='right')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Set y-axis to start from a reasonable value
    min_val = min(baseline_mb) * 0.95
    ax.set_ylim(bottom=min_val)

    fig.text(0.5, 0.02,
             f"GPU: {first_data.get('gpu_name', 'N/A')} | Pre-allocated cache causes higher baseline for v1+",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = Path(output_dir) / 'vram_bar_chart.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    plt.close()


def plot_per_phase_pie_v0_v1(results, output_dir):
    """
    Side-by-side pie charts showing per-phase breakdown for v0 and v1.
    Clearly shows how KV-cache changes the time distribution.
    """
    # Collect per_phase_timing data
    phase_data = {}
    for (benchmark_name, version), data in results.items():
        if benchmark_name == 'per_phase_timing' and version in ('v0', 'v1'):
            if data:
                phase_data[version] = data[-1]  # Latest record

    if 'v0' not in phase_data or 'v1' not in phase_data:
        print("  Need both v0 and v1 per-phase data for pie charts. Skipping.")
        return

    # Define phases and colors
    phase_order = ['embedding', 'attention', 'mlp', 'head', 'other']
    phase_labels = {
        'embedding': 'Embedding',
        'attention': 'Attention',
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, version in enumerate(['v0', 'v1']):
        ax = axes[idx]
        times = phase_data[version]['time_phase_ms']
        total = times.get('total', sum(times.get(p, 0) for p in phase_order))

        # Get values for each phase
        values = []
        labels = []
        colors = []
        for p in phase_order:
            v = times.get(p, 0)
            if v > 0:
                values.append(v)
                pct = (v / total) * 100 if total > 0 else 0
                labels.append(f'{phase_labels[p]}\n{v:.0f}ms ({pct:.1f}%)')
                colors.append(phase_colors[p])

        wedges, texts = ax.pie(values, colors=colors, startangle=90,
                                wedgeprops=dict(width=0.7, edgecolor='white'))

        # Add legend
        ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(0.9, 0.5),
                  fontsize=9, frameon=False)

        cache_label = get_cache_label(version)
        T_star = phase_data[version].get('T_star', '?')
        ax.set_title(f'{cache_label}\nTotal: {total:.0f}ms (T={T_star})',
                     fontweight='bold', fontsize=12)

    # Add overall title
    fig.suptitle('Per-Phase Time Breakdown: v0 vs v1', fontweight='bold', fontsize=14, y=1.02)

    # Add system info
    first_data = phase_data['v0']
    fig.text(0.5, 0.02,
             f"GPU: {first_data.get('gpu_name', 'N/A')} | KV-cache reduces MLP work by avoiding redundant computation",
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = Path(output_dir) / 'per_phase_pie_v0_v1.png'
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
    per_phase_all_data = []  # Collect all per_phase_timing data across versions

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
            per_phase_all_data.extend(data)

    # Plot per_phase_timing with all versions combined
    if per_phase_all_data:
        print("Generating Per-phase Timing plots...")
        plot_per_phase_timing(per_phase_all_data, output_dir)

    # Always emit JSON metrics summary
    print("Writing metrics summary JSON...")
    save_metrics_summary(results, output_dir)

    # Generate comparison plots if multiple versions exist
    print("\nGenerating comparison plots...")
    plot_latency_comparison(results, output_dir)
    plot_vram_comparison(results, output_dir)

    # Split comparison plots: v0 vs v1 (fundamental), then KV variants
    print("\nGenerating split comparison plots (v0 vs v1, KV variants)...")
    plot_v0_vs_v1_latency(results, output_dir)
    plot_kv_variants_latency(results, output_dir)
    plot_speedup_vs_v1(results, output_dir)

    # NEW: Enhanced split plots with projection and bar charts
    print("\nGenerating enhanced split plots (projection, bar charts)...")
    plot_v0_v1_total_latency_with_projection(results, output_dir, T_max=4096)
    plot_kv_variants_per_token_bar(results, output_dir, T_target=1024)
    plot_kv_variants_throughput_bar(results, output_dir, T_target=1024)
    plot_kv_variants_vram_bar(results, output_dir, T_target=1024)
    plot_vram_growth_comparison(results, output_dir)

    # NEW: Improved tradeoff plot at T=1024
    print("\nGenerating improved tradeoff plot (T=1024)...")
    plot_memory_latency_tradeoff_improved(results, output_dir, T_target=1024)

    # Pie chart breakdown for v0 vs v1
    print("\nGenerating per-phase pie charts (v0 vs v1)...")
    plot_per_phase_pie_v0_v1(results, output_dir)
    plot_per_phase_pie_with_percentages(results, output_dir)
    plot_per_phase_breakdown_with_insights(results, output_dir)

    # VRAM bar chart (better than delta plots for papers)
    print("\nGenerating VRAM bar chart...")
    plot_vram_bar_chart(results, output_dir)

    # New comparison plots
    print("\nGenerating additional comparison plots...")
    plot_throughput_comparison(results, output_dir)
    plot_memory_efficiency_comparison(results, output_dir)

    # Plots with 95% confidence intervals
    print("\nGenerating plots with 95% confidence intervals...")
    plot_latency_comparison_with_ci(results, output_dir)
    plot_total_latency_comparison_with_ci(results, output_dir)
    plot_throughput_comparison_with_ci(results, output_dir)

    # Pareto plot (memory vs latency tradeoff)
    print("\nGenerating Pareto plot...")
    plot_pareto_memory_latency(results, output_dir)

    # Perplexity plots
    print("\nGenerating perplexity plots...")
    plot_perplexity_comparison(results, output_dir)
    plot_perplexity_degradation(results, output_dir)

    # Max batch capacity plots
    print("\nGenerating max batch capacity plots...")
    plot_max_batch_capacity(results, output_dir)

    # Export LaTeX tables
    print("\nExporting LaTeX tables...")
    export_latex_tables(results, output_dir)

    # Export perplexity markdown table
    print("\nExporting perplexity markdown table...")
    export_perplexity_markdown_table(results, output_dir)

    print()
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
