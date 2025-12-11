# Cross-Layer KV Sharing: Implementation Plan

## Executive Summary

The current cross-layer sharing implementation is **incomplete**. It degrades output quality but does **not** achieve the intended memory or compute savings. This document explains the current state, ideal state, and exact changes needed.

---

## 1. Current State

### What Cross-Layer Sharing Should Do

The idea: adjacent transformer layers have similar attention patterns. Odd layers can **reuse** K,V from even layers instead of computing their own.

```
Layer 0: Attention(Q0, K0, V0) → stores K0,V0
Layer 1: Attention(Q1, K0, V0) → reuses K0,V0 from layer 0
Layer 2: Attention(Q2, K2, V2) → stores K2,V2
Layer 3: Attention(Q3, K2, V2) → reuses K2,V2 from layer 2
```

**Expected benefits:**
- 50% less KV cache memory (only even layers store)
- Faster inference (odd layers skip K,V computation)

### What Current Code Actually Does

**File:** `model.py`

#### Problem 1: All layers compute K,V (line 72)

```python
def forward(self, x, layer_past=None, use_cache=False, is_cache_owner=True):
    # ...
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # ALWAYS computed, even for odd layers
```

Odd layers compute K,V, then **throw them away** and use cached K,V from even layer.

#### Problem 2: All layers store cache (lines 289-291)

```python
for i, block in enumerate(self.transformer.h):
    if use_cls:
        cache_idx = i if i % 2 == 0 else i - 1
        layer_past = past_key_values[cache_idx]  # odd reads from even ✓
    # ...
    x, present = block(...)
    if use_cache:
        present_key_values.append(present)  # ALL layers store ✗
```

Result: 4-layer model stores `[cache0, cache1, cache2, cache3]` but only `cache0` and `cache2` are ever read.

#### Problem 3: Trimming affects unused caches (lines 471-476)

```python
past_key_values = [
    (k[:, :, -(self.config.block_size - 1):, :],
     v[:, :, -(self.config.block_size - 1):, :])
    for k, v in past_key_values  # trims ALL caches, including unused ones
]
```

### Current State Summary

| Goal | Status | What Happens |
|------|--------|--------------|
| Odd layers use even's K,V for attention | ✅ Works | `layer_past = past_key_values[cache_idx]` |
| Save KV cache memory | ❌ Broken | All layers store cache (odd layers' caches never read) |
| Save K,V computation | ❌ Broken | All layers compute K,V via `self.c_attn(x)` |
| Correct cache trimming | ❌ Broken | Trims unused caches, wastes cycles |

**Net effect:** Quality degrades (odd layers use "stale" K,V) but no resource savings.

---

## 2. Ideal State

### Architecture

```
Layer 0 (owner):    Q0,K0,V0 = c_attn(x)  → Attention(Q0,K0,V0) → store cache[0]
Layer 1 (borrower): Q1 = c_attn_q(x)      → Attention(Q1,K0,V0) → no cache stored
Layer 2 (owner):    Q2,K2,V2 = c_attn(x)  → Attention(Q2,K2,V2) → store cache[1]
Layer 3 (borrower): Q3 = c_attn_q(x)      → Attention(Q3,K2,V2) → no cache stored
```

### Memory Layout

```
Before (current):  present_key_values = [C0, C1, C2, C3]  # 4 caches
After (ideal):     present_key_values = [C0, C2]          # 2 caches

Index mapping:
  Layer 0 → cache[0]    (i // 2 = 0)
  Layer 1 → cache[0]    (i // 2 = 0)
  Layer 2 → cache[1]    (i // 2 = 1)
  Layer 3 → cache[1]    (i // 2 = 1)
```

### Benefits Achieved

| Metric | Current | Ideal | Improvement |
|--------|---------|-------|-------------|
| KV cache entries | n_layer | n_layer / 2 | 50% memory saved |
| K,V computations | n_layer | n_layer / 2 | 50% compute saved |
| Cache trimming ops | n_layer | n_layer / 2 | 50% fewer ops |

---

## 3. Implementation Plan

### Phase 1: Fix Cache Storage (Memory Savings)

**Goal:** Only even layers store cache.

**File:** `model.py`, function `GPT.forward()`, lines 280-291

**Before:**
```python
present_key_values = [] if use_cache else None
for i, block in enumerate(self.transformer.h):
    if use_cls:
        cache_idx = i if i % 2 == 0 else i - 1
        is_cache_owner = (i % 2 == 0)
        layer_past = past_key_values[cache_idx] if past_key_values is not None else None
    else:
        is_cache_owner = True
        layer_past = past_key_values[i] if past_key_values is not None else None
    x, present = block(x, layer_past=layer_past, use_cache=use_cache, is_cache_owner=is_cache_owner)
    if use_cache:
        present_key_values.append(present)
```

**After:**
```python
present_key_values = [] if use_cache else None
for i, block in enumerate(self.transformer.h):
    if use_cls:
        cache_idx = i // 2  # NEW: layer 0,1 → 0; layer 2,3 → 1
        is_cache_owner = (i % 2 == 0)
        layer_past = past_key_values[cache_idx] if past_key_values is not None else None
    else:
        is_cache_owner = True
        layer_past = past_key_values[i] if past_key_values is not None else None
    x, present = block(x, layer_past=layer_past, use_cache=use_cache, is_cache_owner=is_cache_owner)
    if use_cache:
        if not use_cls or is_cache_owner:  # NEW: only even layers store
            present_key_values.append(present)
```

**Also update** `GPT.forward()` past_length calculation (lines 260-264):

**Before:**
```python
if past_key_values is not None:
    if self.config.kv_cache_quant:
        past_length = past_key_values[0][0][0].size(2)
    else:
        past_length = past_key_values[0][0].size(2)
```

**After:** (no change needed - still reads from cache[0])

**Also update** `GPT.generate()` trimming (lines 459-476):

No structural change needed - it will naturally only trim the caches that exist.

---

### Phase 2: Fix Cache Length Check (lines 459-462)

**Before:**
```python
if self.config.kv_cache_quant:
    cache_len = past_key_values[0][0][0].size(2)
else:
    cache_len = past_key_values[0][0].size(2)
```

**After:** (no change needed - cache[0] still exists and has correct length)

---

### Phase 3: Skip K,V Computation for Odd Layers (Compute Savings)

**Goal:** Odd layers only compute Q, not K,V.

**This requires architecture changes.** Two options:

#### Option A: Separate Q Projection (Recommended)

**File:** `model.py`, class `CausalSelfAttention`

**Add new projection in `__init__`:**
```python
def __init__(self, config):
    super().__init__()
    # ... existing code ...
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # NEW: Q-only projection
```

**Modify `forward`:**
```python
def forward(self, x, layer_past=None, use_cache=False, is_cache_owner=True):
    B, T, C = x.size()

    if is_cache_owner:
        # Owner layers: compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    else:
        # Borrower layers: only compute Q
        q = self.c_attn_q(x)
        k, v = None, None  # will use from layer_past

    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # Use past K,V for borrower layers
    if layer_past is not None:
        if self.config.kv_cache_quant:
            (past_key_q, past_key_s), (past_value_q, past_value_s) = layer_past
            past_key = past_key_q.to(q.dtype) * past_key_s
            past_value = past_value_q.to(q.dtype) * past_value_s
        else:
            past_key, past_value = layer_past

        if is_cache_owner:
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        else:
            # Borrower: use past K,V directly (no new K,V to concat)
            k = past_key
            v = past_value

    # ... rest of attention computation ...
```

**Note:** This changes the model architecture. Existing checkpoints won't load without migration.

#### Option B: Reuse Existing Projection, Discard K,V

Simpler but no compute savings:

```python
def forward(self, x, layer_past=None, use_cache=False, is_cache_owner=True):
    B, T, C = x.size()

    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # still compute all
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    if is_cache_owner:
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    else:
        k, v = None, None  # discard computed K,V

    # ... rest uses layer_past for K,V when not owner ...
```

This still computes K,V (wasting compute) but at least doesn't store them.

---

### Phase 4: Update Tests

**File:** `tests/test_kv_cache.py`

Add new tests:

```python
def test_cross_layer_cache_size():
    """Cross-layer sharing should halve cache size."""
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

def test_cross_layer_long_generation():
    """Cross-layer sharing with cache trimming should work."""
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
```

---

## 4. Migration Path

### For Existing Code (No Architecture Change)

Apply Phase 1 only:
- ✅ Fixes memory waste (50% cache reduction)
- ✅ Fixes trimming bug
- ✅ No checkpoint compatibility issues
- ❌ No compute savings

### For Full Benefits (With Architecture Change)

Apply Phase 1 + Phase 3 Option A:
- ✅ 50% cache memory reduction
- ✅ ~33% compute reduction (K,V projection is 2/3 of c_attn)
- ❌ Requires new `c_attn_q` layer
- ❌ Existing checkpoints need migration

**Checkpoint Migration:**
```python
# When loading old checkpoint with cross_layer_sharing:
if 'c_attn_q' not in state_dict:
    # Initialize c_attn_q from c_attn's Q weights
    for i in range(n_layer):
        q_weight = state_dict[f'transformer.h.{i}.attn.c_attn.weight'][:n_embd, :]
        q_bias = state_dict[f'transformer.h.{i}.attn.c_attn.bias'][:n_embd]
        state_dict[f'transformer.h.{i}.attn.c_attn_q.weight'] = q_weight
        state_dict[f'transformer.h.{i}.attn.c_attn_q.bias'] = q_bias
```

---

## 5. Recommendation

### Immediate (Before Merge)

1. **Apply Phase 1** - Fix cache storage to only store even layers
2. **Update tests** - Add cross-layer specific tests
3. **Document limitation** - Note that compute savings require architecture change

### Future (Separate PR)

1. **Apply Phase 3 Option A** - Add separate Q projection
2. **Provide migration script** - For existing checkpoints
3. **Benchmark** - Measure actual memory and compute savings

---

## 6. Summary

| Phase | Change | Memory Savings | Compute Savings | Breaking Change |
|-------|--------|----------------|-----------------|-----------------|
| 1 | Only store even layer caches | ✅ 50% | ❌ None | No |
| 3A | Separate Q projection | ✅ 50% | ✅ ~33% | Yes (new layer) |
| 3B | Discard K,V (no new layer) | ✅ 50% | ❌ None | No |
