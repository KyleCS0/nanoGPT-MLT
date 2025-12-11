# Roofline Analysis: KV Cache Effect

This benchmark evaluates the effect of KV caching on transformer inference by comparing compute vs memory behavior on the roofline model.

## Goal

Demonstrate that:
1. **Prefill** (processing prompt tokens) is similar with/without cache
2. **Decode** (generating new tokens with cache) becomes **memory-bound**

This motivates KV cache optimizations: sharing, quantization, compression.

## Three-Stage Profiling

| Stage | Description | Expected Behavior |
|-------|-------------|-------------------|
| **V0 Prefill** | Forward pass on P tokens, no cache | Baseline compute |
| **V1 Prefill** | Forward pass on P tokens, building cache | Similar to V0 + cache writes |
| **V1 Decode** | Single token decode using cached K/V | Low FLOPs, memory-bound |

## Files

* `profile_decode_step.py` - PyTorch profiling script (three versions)
* `run_ncu.sh` - Runs NCU profiler for all three stages
* `parse_ncu_results.py` - Extracts metrics from NCU reports
* `plot_roofline.py` - Generates publication-quality roofline visualization
* `analyze_results.sh` - End-to-end parsing and plotting

## Plot Features

The roofline plot includes:
- **Publication-quality styling**: 300 DPI, serif fonts, professional grid
- **Efficiency metrics**: Shows % of peak performance and bottleneck type
- **Hardware info footer**: GPU, dtype, batch size, sequence length
- **Multiple formats**: PNG + PDF output for paper inclusion

## Usage

### 1. Run Profiling (requires sudo)

```bash
# Default: P=512 tokens, batch=1, gpt2-medium
sudo bash benchmark/roofline/run_ncu.sh

# Custom parameters
sudo bash benchmark/roofline/run_ncu.sh --P 256 --batch 1 --model gpt2
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--P` | 512 | Prompt length (prefill tokens) |
| `--batch` | 1 | Batch size |
| `--model` | gpt2-medium | Model variant |
| `--dtype` | float16 | Data type |

### 2. Analyze & Plot

```bash
# Must match profiling parameters
bash benchmark/roofline/analyze_results.sh

# Or with custom parameters
bash benchmark/roofline/analyze_results.sh --P 256 --batch 1 --model gpt2
```

**Output:** `benchmark/roofline/roofline.png` and `benchmark/roofline/roofline.pdf`

### Plot Options

`plot_roofline.py` supports additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | A6000 | GPU model: A6000, A100-40GB, A100-80GB, H100-80GB |
| `--dtype` | float16 | Data type for peak FLOPS: float16, float32, bfloat16 |
| `--model` | None | Model name for plot title |
| `--batch-size` | None | Batch size for footer |
| `--seq-length` | None | Sequence length for footer |
| `--formats` | png pdf | Output formats: png, pdf, svg |

**Example:**
```bash
python benchmark/roofline/plot_roofline.py \
    --v0_prefill metrics/v0.json \
    --v1_decode metrics/v1.json \
    --gpu A6000 --model gpt2-medium --formats png pdf
```

## Expected Results

```
V0 Prefill:  High FLOPs, moderate AI  →  Closer to compute-bound
V1 Prefill:  Similar to V0            →  Cache write overhead only
V1 Decode:   Low FLOPs, low AI        →  Memory-bound (key insight!)
```

The decode step being memory-bound means:
- Performance limited by KV cache bandwidth
- Future optimizations should target memory efficiency
- Cross-layer sharing, quantization become valuable

## Version Control

**Commit:** `*.py`, `*.sh`, `README.md`, `roofline.png` (optional)

**Do NOT commit:** `ncu_reports/*.ncu-rep` (100MB+ each), `*_metrics.json`
