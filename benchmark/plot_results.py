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

    # Export LaTeX tables
    print("\nExporting LaTeX tables...")
    export_latex_tables(results, output_dir)

    print()
    print(f"All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
