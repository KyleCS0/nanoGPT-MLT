"""Profiling script for roofline analysis. Run with ncu for detailed metrics."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernels.int8_quant import triton_int8_quant


def theoretical_analysis(M, K, dtype_bytes=2):
    """Calculate theoretical memory traffic and arithmetic intensity."""
    bytes_read = M * K * dtype_bytes
    bytes_written = M * K * 1 + M * 4  # int8 output + fp32 scales
    total_bytes = bytes_read + bytes_written

    # rough flop count: abs, max reduction, div, mul, round, clamp per element
    flops = M * K * 6 + M * 2

    ai = flops / total_bytes
    return {
        'bytes': total_bytes,
        'flops': flops,
        'arithmetic_intensity': ai,
    }


def profile(M=12288, K=64, warmup=10):
    """Profile the kernel. Use with ncu for roofline data."""
    print(f"Profiling M={M}, K={K}")

    theory = theoretical_analysis(M, K)
    print(f"Theoretical: {theory['bytes']/1e6:.2f} MB, AI={theory['arithmetic_intensity']:.2f} FLOPs/B")
    print("Expected: memory-bound (AI << ridge point ~164 for RTX 4090)")

    x = torch.randn(M, K, device='cuda', dtype=torch.float16)

    for _ in range(warmup):
        triton_int8_quant(x)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("triton_int8_quant")
    q, scale = triton_int8_quant(x)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    print(f"Output: q={q.shape}, scale={scale.shape}")


if __name__ == "__main__":
    profile()
