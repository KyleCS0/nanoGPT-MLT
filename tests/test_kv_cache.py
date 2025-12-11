"""
KV-Cache Implementation Tests

Tests to verify the KV-cache implementation is correct:
1. Cached vs non-cached generation produces identical output
2. Training with targets still works (backward compatibility)
3. Cache shapes are correct across all layers
4. Position embeddings are correctly offset
5. Sliding window handles cache overflow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig


def test_cache_correctness():
    """Cached and non-cached generation must produce identical output."""
    print("Test 1: Cache correctness...")

    config = GPTConfig(
        n_layer=4, n_head=4, n_embd=128,
        block_size=256, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    prompt = torch.randint(0, 50257, (1, 10))

    # Generate with cache
    torch.manual_seed(42)
    out_cached = model.generate(prompt.clone(), max_new_tokens=50, use_cache=True)

    # Generate without cache
    torch.manual_seed(42)
    out_no_cache = model.generate(prompt.clone(), max_new_tokens=50, use_cache=False)

    assert torch.equal(out_cached, out_no_cache), "Outputs differ!"
    print("  PASSED")
    return True


def test_training_compat():
    """Training with targets must still work."""
    print("Test 2: Training compatibility...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257
    )
    model = GPT(config)

    X = torch.randint(0, 50257, (2, 32))
    Y = torch.randint(0, 50257, (2, 32))

    logits, loss, cache = model(X, Y)

    assert cache is None, "Cache should be None during training"
    assert loss is not None, "Loss should be computed"
    loss.backward()
    print("  PASSED")
    return True


def test_cache_shapes():
    """Cache shapes must be correct across layers."""
    print("Test 3: Cache shapes...")

    n_layer, n_head, n_embd = 4, 4, 128
    head_dim = n_embd // n_head

    config = GPTConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=256, vocab_size=50257
    )
    model = GPT(config).eval()

    prompt = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        _, _, cache = model(prompt, use_cache=True)

    assert len(cache) == n_layer, f"Expected {n_layer} cache entries, got {len(cache)}"

    for i, (k, v) in enumerate(cache):
        expected = (1, n_head, 10, head_dim)
        assert k.shape == expected, f"Layer {i} K shape wrong: {k.shape} != {expected}"
        assert v.shape == expected, f"Layer {i} V shape wrong: {v.shape} != {expected}"

    print("  PASSED")
    return True


def test_position_offset():
    """Position embeddings must be correctly offset with cache.

    This test verifies that processing tokens one-by-one with cache produces
    the same results as processing all at once without cache.
    """
    print("Test 4: Position offset...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    # Use fixed seed for reproducibility
    torch.manual_seed(123)
    full_seq = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        # Method 1: Process full sequence at once (no cache)
        logits_full, _, _ = model(full_seq, use_cache=False)

        # Method 2: Process first token, then remaining tokens one-by-one with cache
        _, _, cache = model(full_seq[:, :1], use_cache=True)
        for i in range(1, 10):
            logits_cached, _, cache = model(full_seq[:, i:i+1], past_key_values=cache, use_cache=True)

    # Compare final logits (both methods compute last position only when no targets)
    max_diff = (logits_full - logits_cached).abs().max().item()

    assert max_diff < 1e-5, f"Logits differ by {max_diff} - position offset may be incorrect"
    print("  PASSED")
    return True


def test_sliding_window():
    """Generation must work when exceeding block_size."""
    print("Test 5: Sliding window...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=32, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    prompt = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=40, use_cache=True)

    assert out.shape[1] == 50, f"Expected 50 tokens, got {out.shape[1]}"
    print("  PASSED")
    return True


def test_incremental_cache_growth():
    """Cache should grow by one position per token generated."""
    print("Test 6: Incremental cache growth...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    # Start with a prompt
    prompt = torch.randint(0, 50257, (1, 5))

    with torch.no_grad():
        # First forward: process prompt
        _, _, cache = model(prompt, use_cache=True)
        assert cache[0][0].size(2) == 5, f"Initial cache should have 5 positions"

        # Second forward: add one token
        next_token = torch.randint(0, 50257, (1, 1))
        _, _, cache = model(next_token, past_key_values=cache, use_cache=True)
        assert cache[0][0].size(2) == 6, f"Cache should have 6 positions after one token"

        # Third forward: add another token
        next_token = torch.randint(0, 50257, (1, 1))
        _, _, cache = model(next_token, past_key_values=cache, use_cache=True)
        assert cache[0][0].size(2) == 7, f"Cache should have 7 positions after two tokens"

    print("  PASSED")
    return True


def test_batch_generation():
    """KV-cache must work correctly with batch_size > 1."""
    print("Test 7: Batch generation...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    # Batch of 3 different prompts
    prompts = torch.randint(0, 50257, (3, 8))

    torch.manual_seed(42)
    out_cached = model.generate(prompts.clone(), max_new_tokens=20, use_cache=True)

    torch.manual_seed(42)
    out_no_cache = model.generate(prompts.clone(), max_new_tokens=20, use_cache=False)

    assert torch.equal(out_cached, out_no_cache), "Batch outputs differ!"
    assert out_cached.shape == (3, 28), f"Expected shape (3, 28), got {out_cached.shape}"
    print("  PASSED")
    return True


def test_determinism():
    """Same seed must produce identical results across runs."""
    print("Test 8: Determinism...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    prompt = torch.randint(0, 50257, (1, 5))

    # Run twice with same seed
    torch.manual_seed(999)
    out1 = model.generate(prompt.clone(), max_new_tokens=30, use_cache=True)

    torch.manual_seed(999)
    out2 = model.generate(prompt.clone(), max_new_tokens=30, use_cache=True)

    assert torch.equal(out1, out2), "Non-deterministic outputs!"
    print("  PASSED")
    return True


def test_long_generation():
    """Test generation of many tokens (stress test)."""
    print("Test 9: Long generation (100 tokens)...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=256, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    prompt = torch.randint(0, 50257, (1, 10))

    torch.manual_seed(42)
    out_cached = model.generate(prompt.clone(), max_new_tokens=100, use_cache=True)

    torch.manual_seed(42)
    out_no_cache = model.generate(prompt.clone(), max_new_tokens=100, use_cache=False)

    assert torch.equal(out_cached, out_no_cache), "Long generation outputs differ!"
    assert out_cached.shape[1] == 110, f"Expected 110 tokens, got {out_cached.shape[1]}"
    print("  PASSED")
    return True


def test_single_token_prompt():
    """Generation must work with minimal prompt (single token)."""
    print("Test 10: Single token prompt...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    # Single token prompt
    prompt = torch.randint(0, 50257, (1, 1))

    torch.manual_seed(42)
    out_cached = model.generate(prompt.clone(), max_new_tokens=20, use_cache=True)

    torch.manual_seed(42)
    out_no_cache = model.generate(prompt.clone(), max_new_tokens=20, use_cache=False)

    assert torch.equal(out_cached, out_no_cache), "Single token prompt outputs differ!"
    assert out_cached.shape[1] == 21, f"Expected 21 tokens, got {out_cached.shape[1]}"
    print("  PASSED")
    return True


def test_cache_reuse_across_calls():
    """Cache from one forward can be reused in subsequent single-token calls."""
    print("Test 11: Cache reuse across calls...")

    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        block_size=128, vocab_size=50257, dropout=0.0
    )
    model = GPT(config).eval()

    torch.manual_seed(456)
    full_seq = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        # Method 1: Process all at once
        logits_full, _, _ = model(full_seq, use_cache=False)

        # Method 2: Process prompt, then continue token-by-token
        _, _, cache = model(full_seq[:, :5], use_cache=True)
        for i in range(5, 10):
            logits_cached, _, cache = model(full_seq[:, i:i+1], past_key_values=cache, use_cache=True)

    # Compare final logits
    max_diff = (logits_full - logits_cached).abs().max().item()

    assert max_diff < 1e-5, f"Cache reuse failed, diff = {max_diff}"
    print("  PASSED")
    return True

def test_grand_slam_cache_types():
    """Verify cache types for combined quantization and cross-layer sharing."""
    print("Test 12: Grand Slam cache types...")

    config = GPTConfig(
        n_layer=4, n_head=4, n_embd=128,
        block_size=256, vocab_size=50257, dropout=0.0,
        cross_layer_sharing=True, kv_cache_quant=True
    )
    model = GPT(config).eval()

    prompt = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        _, _, cache = model(prompt, use_cache=True, cross_layer_sharing=True)

    # With cross-layer sharing, only owner layers (even) store cache
    expected_cache_len = config.n_layer // 2
    assert len(cache) == expected_cache_len, f"Expected {expected_cache_len} cache entries, got {len(cache)}"

    # All stored caches should be from owner layers and should be quantized
    for i, (k, v) in enumerate(cache):
        # Owner (even) layers should have quantized cache: ((tensor, tensor), (tensor,tensor))
        assert isinstance(k, tuple) and len(k) == 2, f"Cache {i} K should be a tuple (quantized)"
        assert isinstance(v, tuple) and len(v) == 2, f"Cache {i} V should be a tuple (quantized)"
        assert isinstance(k[0], torch.Tensor) and k[0].dtype == torch.int8, f"Cache {i} K[0] should be int8 tensor"
        assert isinstance(v[0], torch.Tensor) and v[0].dtype == torch.int8, f"Cache {i} V[0] should be int8 tensor"

    print("  PASSED")
    return True


def test_cross_layer_cache_size():
    """Cross-layer sharing should halve cache size."""
    print("Test 13: Cross-layer sharing cache size...")
    config = GPTConfig(
        n_layer=4, n_head=4, n_embd=128,
        block_size=256, vocab_size=50257, dropout=0.0,
        cross_layer_sharing=True
    )
    model = GPT(config).eval()
    prompt = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        _, _, cache = model(prompt, use_cache=True)

    # Should only have 2 cache entries (layers 0 and 2), not 4
    assert len(cache) == config.n_layer // 2, f"Expected {config.n_layer // 2} caches, got {len(cache)}"
    print("  PASSED")
    return True

def test_cross_layer_long_generation():
    """Cross-layer sharing with cache trimming should work."""
    print("Test 14: Cross-layer sharing long generation...")
    config = GPTConfig(
        n_layer=4, n_head=4, n_embd=128,
        block_size=32, vocab_size=50257, dropout=0.0,
        cross_layer_sharing=True
    )
    model = GPT(config).eval()
    prompt = torch.randint(0, 50257, (1, 10))

    # Generate beyond block_size to trigger trimming
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=40, use_cache=True)

    assert out.shape[1] == 50, f"Expected 50 tokens, got {out.shape[1]}"
    print("  PASSED")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("KV-Cache Tests")
    print("=" * 50)

    tests = [
        test_cache_correctness,
        test_training_compat,
        test_cache_shapes,
        test_position_offset,
        test_sliding_window,
        test_incremental_cache_growth,
        test_batch_generation,
        test_determinism,
        test_long_generation,
        test_single_token_prompt,
        test_cache_reuse_across_calls,
        test_grand_slam_cache_types,
        test_cross_layer_cache_size,
        test_cross_layer_long_generation,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} passed")
    exit(0 if failed == 0 else 1)
