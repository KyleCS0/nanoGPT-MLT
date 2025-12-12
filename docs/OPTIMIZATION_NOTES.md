# KV-Cache Optimization Notes

Technical notes on KV-cache implementation, known issues, and optimization strategies.

---

## Current Implementation Status

| Version | Description | Status |
|---------|-------------|--------|
| v0 | No cache (baseline) | Working |
| v1 | KV-cache enabled | Working |
| v2 | KV-cache + INT8 quantization | Working |
| v3 | KV-cache + cross-layer sharing | Working |
| v4 | KV-cache + INT8 + cross-layer | Working |

---

## Known Performance Issues

### Issue: Cache Overhead Exceeds Computation Savings

**Symptom**: v1 (with cache) is ~5% slower than v0 (no cache) for small batches and moderate sequences.

**Root Cause**: Memory allocation overhead from `torch.cat()` exceeds computation savings.

```python
# Current hot path (expensive):
k = torch.cat([past_key, k], dim=2)  # GPU allocation every iteration
v = torch.cat([past_value, v], dim=2)
```

**Impact**:
- 12 layers × T iterations = 12T GPU allocations per generation
- For T=1024: 12,288 allocations

### Recommended Fix: Circular Buffer

Replace dynamic concatenation with pre-allocated circular buffer:

```python
# Pre-allocate once:
cache_k = torch.empty(B, nh, block_size, hs, device=device)
cache_v = torch.empty(B, nh, block_size, hs, device=device)

# In-place update (no allocation):
pos = cache_position % block_size
cache_k[:, :, pos, :] = new_k[:, :, -1, :]
cache_v[:, :, pos, :] = new_v[:, :, -1, :]
```

**Expected improvement**: 3-8x speedup after optimization.

---

## Cross-Layer Sharing Implementation

### How It Works

Adjacent layers share KV cache:
- Layer 0 (owner): Computes Q, K, V → stores cache
- Layer 1 (borrower): Computes Q only → uses Layer 0's K, V
- Layer 2 (owner): Computes Q, K, V → stores cache
- Layer 3 (borrower): Computes Q only → uses Layer 2's K, V

### Current Limitation

All layers still compute K, V via `c_attn(x)`, then borrowers discard their K, V.

**Memory savings**: 50% (only owners store cache)
**Compute savings**: None currently (would require separate Q projection)

### Future Enhancement

Add separate Q-only projection for borrower layers:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Q, K, V
        self.c_attn_q = nn.Linear(n_embd, n_embd)    # Q only (for borrowers)
```

---

## INT8 Quantization

### Implementation

```python
# Quantize (in forward pass):
k_absmax = k.abs().max()
k_scale = k_absmax / 127.0 if k_absmax > 0 else 1.0
k_quant = (k / k_scale).round().clamp(-127, 127).to(torch.int8)

# Dequantize (when reading cache):
k = k_quant.to(q.dtype) * k_scale
```

### Performance Note

Dequantization of entire cache every step adds overhead. Future optimization: fused CUDA kernels for INT8 attention.

---

## Optimization Priority List

| Priority | Optimization | Impact | Complexity |
|----------|-------------|--------|------------|
| 1 | Pre-allocate KV cache buffers | High | Medium |
| 2 | Circular buffer (eliminate torch.cat) | High | Medium |
| 3 | Pre-allocate output tensor in generate() | Medium | Low |
| 4 | Fused INT8 dequant kernels | Medium | High |
| 5 | Separate Q projection for cross-layer | Low | Medium |

---

## Benchmark Configuration

Default settings in `benchmark/config.yaml`:
- `num_warmup_runs`: 10 (for stable GPU state)
- `num_measure_runs`: 20 (for statistical significance)
- `T_values`: [32, 64, ..., 1024]

For GPU clock stability, use:
```bash
./benchmark/activate_clock_lock.sh   # Lock clocks before benchmark
./benchmark/deactivate_clock_lock.sh # Restore after
```

---

## Testing Cache Correctness

Key tests in `tests/test_kv_cache.py`:

1. **Cache shape validation**: Verify (B, H, T, HS) dimensions
2. **Output equivalence**: `generate(use_cache=True)` == `generate(use_cache=False)`
3. **Long generation**: Test cache trimming when T > block_size
4. **Cross-layer**: Verify only n_layer/2 caches stored

Run tests:
```bash
python tests/test_kv_cache.py
```

---

## References

- Flash Attention: https://github.com/Dao-AILab/flash-attention
- HuggingFace KV Cache: https://huggingface.co/docs/transformers/kv_cache
- NVIDIA NSight Systems: For profiling GPU kernels
