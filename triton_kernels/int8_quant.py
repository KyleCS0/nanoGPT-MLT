"""INT8 quantization kernels for KV-cache."""

import torch
import triton
import triton.language as tl
import time


def pytorch_int8_quant_per_row(x):
    """Per-row INT8 quantization. Returns (q, scale)."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def pytorch_int8_quant_per_tensor(x):
    """Per-tensor INT8 quantization. Returns (q, scale)."""
    amax = x.abs().max()
    scale = amax / 127.0 if amax > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


@triton.jit
def _quant_kernel(
    x_ptr, q_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_qm, stride_qk,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K

    x = tl.load(x_ptr + row * stride_xm + offs_k * stride_xk, mask=mask, other=0.0)

    amax = tl.max(tl.abs(x))
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    inv_scale = tl.where(amax > 0, 127.0 / amax, 0.0)

    tl.store(scale_ptr + row, scale)

    q = x.to(tl.float32) * inv_scale
    q = tl.floor(q + 0.5)
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)

    tl.store(q_ptr + row * stride_qm + offs_k * stride_qk, q.to(tl.int8), mask=mask)


def triton_int8_quant(x):
    """Fused per-row INT8 quantization. Returns (q, scale)."""
    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    K = x.shape[-1]
    x_2d = x.view(-1, K)
    M = x_2d.shape[0]

    q = torch.empty_like(x_2d, dtype=torch.int8)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_K = max(min(BLOCK_K, 1024), 32)

    _quant_kernel[(M,)](
        x_2d, q, scale,
        M, K,
        x_2d.stride(0), x_2d.stride(1),
        q.stride(0), q.stride(1),
        BLOCK_K=BLOCK_K,
    )

    return q.view(orig_shape), scale.view(orig_shape[:-1])


def dequantize(q, scale):
    """Dequantize INT8 back to float."""
    return q.to(torch.float32) * scale.unsqueeze(-1)


def test_correctness(verbose=True):
    """Test Triton kernel against PyTorch baseline."""
    cases = [
        (1, 64), (128, 64), (12288, 64),
        (1, 128), (32768, 64), (100, 100),
    ]

    all_pass = True
    for M, K in cases:
        x = torch.randn(M, K, device='cuda', dtype=torch.float16)

        q_pt, s_pt = pytorch_int8_quant_per_row(x)
        q_tr, s_tr = triton_int8_quant(x)

        x_pt = dequantize(q_pt, s_pt)
        x_tr = dequantize(q_tr, s_tr)

        err = (x_pt - x_tr).abs().max().item()
        ok = err < 0.1
        all_pass = all_pass and ok

        if verbose:
            print(f"M={M}, K={K}: {'PASS' if ok else 'FAIL'} (err={err:.4f})")

    # edge cases
    x_zeros = torch.zeros(128, 64, device='cuda', dtype=torch.float16)
    q, _ = triton_int8_quant(x_zeros)
    all_pass = all_pass and (q == 0).all().item()
    if verbose:
        print(f"zeros: {'PASS' if (q == 0).all() else 'FAIL'}")

    return all_pass


def benchmark(warmup=10, iters=100, verbose=True):
    """Benchmark PyTorch vs Triton."""
    configs = [
        (12288, 64, "GPT-2 KV-cache"),
        (24576, 64, "GPT-2 batch=2"),
        (16384, 64, "GPT-2-medium"),
        (32768, 64, "large"),
        (4096, 128, "head_dim=128"),
    ]

    results = []
    for M, K, desc in configs:
        x = torch.randn(M, K, device='cuda', dtype=torch.float16)

        for _ in range(warmup):
            pytorch_int8_quant_per_row(x)
            triton_int8_quant(x)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            pytorch_int8_quant_per_row(x)
        torch.cuda.synchronize()
        pt_ms = (time.perf_counter() - t0) / iters * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            triton_int8_quant(x)
        torch.cuda.synchronize()
        tr_ms = (time.perf_counter() - t0) / iters * 1000

        speedup = pt_ms / tr_ms
        results.append((desc, M, K, pt_ms, tr_ms, speedup))

        if verbose:
            print(f"{desc}: PyTorch={pt_ms:.3f}ms, Triton={tr_ms:.3f}ms, {speedup:.1f}x")

    return results


if __name__ == "__main__":
    print("=== Correctness ===")
    test_correctness()
    print("\n=== Benchmark ===")
    benchmark()
