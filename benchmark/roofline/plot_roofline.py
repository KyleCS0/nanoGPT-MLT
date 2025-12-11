"""
Generate roofline diagram from ncu profiling results.

Usage:
    python plot_roofline.py --v0 metrics_v0.json --v1 metrics_v1.json
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# A6000 specs
A6000_PEAK_FLOPS = 38.7e12    # FP16: 38.7 TFLOPS
A6000_PEAK_BW = 768e9         # 768 GB/s
A6000_RIDGE = A6000_PEAK_FLOPS / A6000_PEAK_BW  # ~50.4 FLOPs/Byte

def plot_roofline(v0_metrics, v1_metrics, output_path, title="Roofline Analysis: V0 vs V1"):
    """Generate roofline plot."""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline ceiling
    ai_range = np.logspace(-1, 3, 500)
    roofline = np.minimum(A6000_PEAK_FLOPS, ai_range * A6000_PEAK_BW)

    ax.loglog(ai_range, roofline, 'b-', linewidth=2.5, label='A6000 Roofline')

    # Ridge point
    ax.axvline(A6000_RIDGE, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(A6000_RIDGE * 1.1, A6000_PEAK_FLOPS * 0.5, f'Ridge\n({A6000_RIDGE:.1f})',
            fontsize=9, color='gray')

    # Plot V0
    if v0_metrics:
        ax.scatter([v0_metrics['arithmetic_intensity']], [v0_metrics['achieved_flops']],
                   s=200, c='red', marker='o', label=f"V0 (no cache)\nAI={v0_metrics['arithmetic_intensity']:.1f}",
                   zorder=5, edgecolors='black', linewidths=1)

    # Plot V1
    if v1_metrics:
        ax.scatter([v1_metrics['arithmetic_intensity']], [v1_metrics['achieved_flops']],
                   s=200, c='green', marker='s', label=f"V1 (KV-cache)\nAI={v1_metrics['arithmetic_intensity']:.1f}",
                   zorder=5, edgecolors='black', linewidths=1)

    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Performance (FLOP/s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.1, 1000])
    ax.set_ylim([1e9, 1e14])

    # Add region labels
    ax.text(1, 1e13, 'Memory\nBound', fontsize=11, ha='center', color='blue', alpha=0.7)
    ax.text(200, 1e13, 'Compute\nBound', fontsize=11, ha='center', color='blue', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v0', type=str, help='V0 metrics JSON file')
    parser.add_argument('--v1', type=str, help='V1 metrics JSON file')
    parser.add_argument('--output', type=str, default='benchmark/roofline/roofline.png')
    args = parser.parse_args()

    v0_metrics = json.load(open(args.v0)) if args.v0 else None
    v1_metrics = json.load(open(args.v1)) if args.v1 else None

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_roofline(v0_metrics, v1_metrics, args.output)

if __name__ == '__main__':
    main()
