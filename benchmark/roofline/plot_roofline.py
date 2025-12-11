#!/usr/bin/env python3
"""
Generate publication-quality roofline diagram for KV Cache analysis.

Three-stage comparison:
  - V0 Prefill: Full forward, no cache (baseline)
  - V1 Prefill: Full forward, building cache
  - V1 Decode:  Single token with cached K/V (key comparison point)

Usage:
    python plot_roofline.py --v0_prefill m0.json --v1_prefill m1.json --v1_decode m2.json
    python plot_roofline.py --v0_prefill m0.json --v1_decode m2.json --model gpt2-medium
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality style (matching plot_results.py)
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
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.grid': True,
})

# GPU specifications database
GPU_SPECS = {
    'A6000': {
        'peak_fp16_tflops': 38.7,
        'peak_fp32_tflops': 38.7,
        'dram_bandwidth_gb_s': 768,
        'name': 'NVIDIA RTX A6000',
    },
    'A100-40GB': {
        'peak_fp16_tflops': 312,
        'peak_fp32_tflops': 156,
        'dram_bandwidth_gb_s': 1555,
        'name': 'NVIDIA A100 40GB',
    },
    'A100-80GB': {
        'peak_fp16_tflops': 312,
        'peak_fp32_tflops': 156,
        'dram_bandwidth_gb_s': 2039,
        'name': 'NVIDIA A100 80GB',
    },
    'H100-80GB': {
        'peak_fp16_tflops': 989,
        'peak_fp32_tflops': 495,
        'dram_bandwidth_gb_s': 3350,
        'name': 'NVIDIA H100 80GB',
    },
}

# Color palette (matching plot_results.py)
POINT_COLORS = {
    'v0_prefill': '#E63946',  # Red - no cache
    'v1_prefill': '#457B9D',  # Blue - building cache
    'v1_decode': '#2A9D8F',   # Teal - using cache
}

POINT_MARKERS = {
    'v0_prefill': 'o',
    'v1_prefill': 's',
    'v1_decode': '^',
}


def calculate_efficiency(ai, achieved_flops, peak_flops, bandwidth):
    """Calculate efficiency metrics for a data point."""
    ridge_point = peak_flops / bandwidth
    roofline_perf = min(peak_flops, ai * bandwidth)

    # Efficiency relative to roofline ceiling
    efficiency_pct = (achieved_flops / roofline_perf) * 100 if roofline_perf > 0 else 0

    # Determine bottleneck
    bottleneck = 'memory-bound' if ai < ridge_point else 'compute-bound'

    return {
        'efficiency_pct': min(efficiency_pct, 100),  # Cap at 100%
        'bottleneck': bottleneck,
        'roofline_perf': roofline_perf,
        'ridge_point': ridge_point,
    }


def plot_roofline(v0_prefill, v1_prefill, v1_decode, output_path,
                  gpu='A6000', dtype='float16', model_name=None,
                  batch_size=None, seq_length=None, output_formats=None):
    """Generate publication-quality roofline plot."""

    if output_formats is None:
        output_formats = ['png', 'pdf']

    # Get GPU specs
    specs = GPU_SPECS.get(gpu, GPU_SPECS['A6000'])
    dtype_key = 'peak_fp16_tflops' if dtype in ['float16', 'bfloat16'] else 'peak_fp32_tflops'
    peak_flops = specs[dtype_key] * 1e12  # Convert to FLOP/s
    dram_bw = specs['dram_bandwidth_gb_s'] * 1e9  # Convert to B/s
    ridge_point = peak_flops / dram_bw

    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline ceiling
    ai_range = np.logspace(-2, 3, 500)
    roofline = np.minimum(peak_flops, ai_range * dram_bw)

    ax.loglog(ai_range, roofline, '-', linewidth=2.5, color='#2E86AB',
              label=f'{specs["name"]} Roofline ({dtype.upper()})')

    # Ridge point annotation
    ax.axvline(ridge_point, color='#AAAAAA', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(ridge_point * 1.15, peak_flops * 0.12,
            f'Ridge\n({ridge_point:.1f} FLOP/B)',
            fontsize=9, color='#666666', ha='left')

    # Region labels
    ax.text(0.15, peak_flops * 0.25, 'Memory\nBound',
            fontsize=11, ha='center', color='#666666', alpha=0.7, style='italic')
    ax.text(ridge_point * 4, peak_flops * 0.25, 'Compute\nBound',
            fontsize=11, ha='center', color='#666666', alpha=0.7, style='italic')

    # Plot data points
    markers_data = []
    data_points = [
        ('V0 Prefill (no cache)', v0_prefill, 'v0_prefill'),
        ('V1 Prefill (build cache)', v1_prefill, 'v1_prefill'),
        ('V1 Decode (use cache)', v1_decode, 'v1_decode'),
    ]

    for name, data, key in data_points:
        if data and 'arithmetic_intensity' in data:
            ai = data['arithmetic_intensity']
            perf = data['achieved_flops']

            # Calculate efficiency
            eff = calculate_efficiency(ai, perf, peak_flops, dram_bw)

            # Plot point with alpha for overlapping visibility
            ax.scatter([ai], [perf], s=200, c=POINT_COLORS[key],
                       marker=POINT_MARKERS[key], zorder=5, alpha=0.6,
                       edgecolors='black', linewidths=1.2)

            markers_data.append({
                'name': name,
                'color': POINT_COLORS[key],
                'marker': POINT_MARKERS[key],
                'ai': ai,
                'perf': perf,
                'efficiency': eff,
            })

    # Create rich legend with metrics
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    for m in markers_data:
        handle = Line2D([0], [0], marker=m['marker'], color='w',
                        markerfacecolor=m['color'], markersize=10,
                        markeredgecolor='black', markeredgewidth=1)
        legend_handles.append(handle)

        # Multi-line label with metrics
        label = (f"{m['name']}\n"
                 f"  AI={m['ai']:.2f} FLOP/B | {m['perf']/1e12:.2f} TFLOP/s\n"
                 f"  {m['efficiency']['efficiency_pct']:.0f}% eff. ({m['efficiency']['bottleneck']})")
        legend_labels.append(label)

    # Add roofline to legend
    legend_handles.append(Line2D([0], [0], color='#2E86AB', linewidth=2.5))
    legend_labels.append(f'Peak: {peak_flops/1e12:.1f} TFLOP/s | BW: {dram_bw/1e9:.0f} GB/s')

    ax.legend(legend_handles, legend_labels, loc='lower right',
              fontsize=8, framealpha=0.95, fancybox=True,
              borderpad=1, labelspacing=1.5)

    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontweight='bold')
    ax.set_ylabel('Performance (FLOP/s)', fontweight='bold')

    # Dynamic title
    title = 'Roofline Analysis: KV Cache Effect'
    if model_name:
        title += f' ({model_name})'
    ax.set_title(title, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_xlim([0.01, 1000])
    ax.set_ylim([1e9, peak_flops * 2])

    # Add key insight annotation for decode step
    if v1_decode and 'arithmetic_intensity' in v1_decode:
        ai_decode = v1_decode['arithmetic_intensity']
        perf_decode = v1_decode['achieved_flops']
        ax.annotate('Decode step is\nmemory-bound',
                    xy=(ai_decode, perf_decode),
                    xytext=(ai_decode * 8, perf_decode * 4),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='#666666', alpha=0.7))

    # Hardware info footer
    footer_parts = [f"GPU: {specs['name']}"]
    footer_parts.append(f"dtype: {dtype}")
    if batch_size is not None:
        footer_parts.append(f"Batch: {batch_size}")
    if seq_length is not None:
        footer_parts.append(f"Seq: {seq_length}")

    fig.text(0.5, 0.02, ' | '.join(footer_parts),
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save in multiple formats
    output_base = Path(output_path).stem
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in output_formats:
        out_file = output_dir / f"{output_base}.{fmt}"
        if fmt == 'pdf':
            fig.savefig(out_file, format='pdf', bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        else:
            fig.savefig(out_file, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        print(f"Saved: {out_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality roofline plot for KV cache analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_roofline.py --v0_prefill m0.json --v1_decode m2.json
  python plot_roofline.py --v0_prefill m0.json --gpu A100-40GB --model gpt2-medium
  python plot_roofline.py --v0_prefill m0.json --formats png pdf
        """
    )

    # Data inputs
    parser.add_argument('--v0_prefill', type=str, help='V0 prefill metrics JSON')
    parser.add_argument('--v1_prefill', type=str, help='V1 prefill metrics JSON')
    parser.add_argument('--v1_decode', type=str, help='V1 decode metrics JSON')

    # Legacy support
    parser.add_argument('--v0', type=str, help='(Legacy) V0 metrics JSON')
    parser.add_argument('--v1', type=str, help='(Legacy) V1 metrics JSON')

    # Output configuration
    parser.add_argument('--output', type=str, default='benchmark/roofline/roofline.png',
                        help='Output file path (base name for multiple formats)')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                        choices=['png', 'pdf', 'svg'],
                        help='Output formats (default: png pdf)')

    # Hardware configuration
    parser.add_argument('--gpu', type=str, default='A6000',
                        choices=list(GPU_SPECS.keys()),
                        help='GPU model for roofline specs (default: A6000)')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Data type for peak FLOPS calculation (default: float16)')

    # Model/experiment metadata
    parser.add_argument('--model', type=str, default=None,
                        help='Model name for plot title (e.g., gpt2-medium)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for footer')
    parser.add_argument('--seq-length', type=int, default=None,
                        help='Sequence length for footer')

    args = parser.parse_args()

    # Load metrics
    v0_prefill = json.load(open(args.v0_prefill)) if args.v0_prefill else None
    v1_prefill = json.load(open(args.v1_prefill)) if args.v1_prefill else None
    v1_decode = json.load(open(args.v1_decode)) if args.v1_decode else None

    # Legacy fallback
    if v0_prefill is None and args.v0:
        v0_prefill = json.load(open(args.v0))
    if v1_decode is None and args.v1:
        v1_decode = json.load(open(args.v1))

    plot_roofline(
        v0_prefill, v1_prefill, v1_decode, args.output,
        gpu=args.gpu,
        dtype=args.dtype,
        model_name=args.model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        output_formats=args.formats,
    )


if __name__ == '__main__':
    main()
