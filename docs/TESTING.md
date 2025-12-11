# Testing Guide

## KV-Cache Implementation Tests

### 1. Unit Tests (Required)

Run the comprehensive test suite to verify KV-cache correctness:

```bash
python tests/test_kv_cache.py
```

**Expected output**: All 11 tests pass

| Test | Description |
|------|-------------|
| 1 | Cache vs non-cache produces identical output |
| 2 | Training with targets still works |
| 3 | Cache shapes correct across layers |
| 4 | Position embeddings correctly offset |
| 5 | Sliding window handles cache overflow |
| 6 | Cache grows incrementally per token |
| 7 | Batch generation (batch_size > 1) |
| 8 | Determinism with same seed |
| 9 | Long generation (100 tokens) |
| 10 | Single token prompt |
| 11 | Cache reuse across calls |

---

### 2. Sample Generation

Test text generation with pretrained GPT-2:

```bash
# With KV-cache (default)
python sample.py --init_from=gpt2 --num_samples=1 --max_new_tokens=50

# Without KV-cache (for comparison)
python sample.py --init_from=gpt2 --num_samples=1 --max_new_tokens=50 --use_cache=False
```

**Expected**: Both commands generate coherent text without errors.

---

### 3. Capacity Stress Test

Test maximum batch size for inference:

```bash
# With KV-cache (v1)
python capacity_test/bench_capacity_inference.py --version v1

# Without KV-cache (v0 baseline)
python capacity_test/bench_capacity_inference.py --version v0
```

**Expected**: v0 supports higher batch sizes (no cache memory overhead), v1 supports lower batch sizes but faster per-token generation.

---

### 4. Latency Benchmark

Compare generation latency between versions:

```bash
# Run both versions
python benchmark/main.py --config benchmark/config.yaml latency_vs_T --version v0 v1

# Run only v0 (baseline)
python benchmark/main.py --config benchmark/config.yaml latency_vs_T --version v0

# Run only v1 (KV-cache)
python benchmark/main.py --config benchmark/config.yaml latency_vs_T --version v1
```

Results are logged to `benchmark/results.jsonl`.

---

### 5. VRAM Benchmark

Compare memory usage between versions:

```bash
python benchmark/main.py --config benchmark/config.yaml vram_vs_T --version v0 v1
```

---

### 6. Quick Latency Test (Python)

For a quick test without full benchmark:

```python
python -c "
import yaml
from benchmark.main import run_latency_vs_T, load_model, create_prompt

with open('benchmark/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['versions'] = ['v0', 'v1']
config['T_values'] = [32, 64]
config['num_warmup_runs'] = 1
config['num_measure_runs'] = 3

run_latency_vs_T(config)
"
```

---

## Version Reference

| Version | Description | `use_cache` |
|---------|-------------|-------------|
| v0 | No cache (baseline) | `False` |
| v1 | KV-cache enabled | `True` |

---

## Troubleshooting

### "No such file: out/ckpt.pt"
Use `--init_from=gpt2` to load pretrained weights from HuggingFace instead of a local checkpoint.

### OOM errors during benchmark
Reduce `T_values` in `benchmark/config.yaml` or use a smaller model preset.
