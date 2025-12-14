# nanoGPT Benchmarking System Documentation

A comprehensive guide to the KV-cache optimization benchmarking system.

## Quick Start

```bash
# Run all benchmarks comparing v0 (no cache) vs v1 (with cache)
python benchmark/main.py --version v0 v1

# Run only latency benchmark on all versions
python benchmark/main.py latency --version v0 v1 v2 v3 v4

# Run VRAM benchmark on memory-optimized versions with larger model
python benchmark/main.py vram --version v1 v2 v3 v4 --preset gpt2-medium

# Run perplexity evaluation
python benchmark/perplexity.py --version v0 v1 v2 v3 v4

# Generate plots from results
python benchmark/plot_results.py
```

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Version System](#version-system)
4. [Benchmarks](#benchmarks)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Results & Analysis](#results--analysis)
8. [Extending the System](#extending-the-system)

---

## Overview

The benchmarking system measures performance and memory characteristics of different KV-cache optimization strategies for GPT models. It provides:

- **Latency benchmarks**: Token generation speed across different sequence lengths
- **VRAM benchmarks**: Memory usage patterns during generation
- **Per-phase timing**: Breakdown of time spent in embedding, attention, MLP, and head layers
- **Perplexity evaluation**: Model quality verification on WikiText-2
- **Roofline analysis**: Hardware efficiency profiling (FLOPs vs memory bandwidth)

### Key Design Principles

1. **Single Source of Truth**: All version definitions live in `benchmark/versions.py`
2. **Modular Benchmarks**: Each benchmark can run independently or together
3. **Reproducible Results**: Seeded random generation, JSONL logging with full metadata
4. **Extensible**: Easy to add new versions or benchmarks

---

## Architecture

```
benchmark/
├── versions.py          # Version registry (SINGLE SOURCE OF TRUTH)
├── main.py              # Main benchmark runner (latency, vram, phase)
├── perplexity.py        # Model quality evaluation
├── config.yaml          # Benchmark configuration
├── results.jsonl        # Output results (JSONL format)
├── plot_results.py      # Visualization tools
└── roofline/
    ├── profile_decode_step.py   # NCU profiling script
    ├── run_ncu.sh               # NCU automation
    ├── parse_ncu_results.py     # Extract metrics from NCU
    └── plot_roofline.py         # Roofline visualization
```

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  config.yaml    │────▶│   main.py        │────▶│  results.jsonl  │
│  (parameters)   │     │   (benchmarks)   │     │  (raw data)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                         │
                                ▼                         ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  versions.py     │     │ plot_results.py │
                        │  (model config)  │     │ (visualization) │
                        └──────────────────┘     └─────────────────┘
```

---

## Version System

### Available Versions

| Version | Description | use_cache | kv_cache_quant | cross_layer_sharing |
|---------|-------------|-----------|----------------|---------------------|
| **v0** | No cache (baseline) | False | False | False |
| **v1** | KV-cache enabled | True | False | False |
| **v2** | KV-cache + INT8 quantization | True | True | False |
| **v3** | KV-cache + cross-layer sharing | True | False | True |
| **v4** | KV-cache + INT8 + cross-layer sharing | True | True | True |

### How Versions Work

Each version modifies model behavior through three flags:

1. **`use_cache`**: Whether to use KV-cache during generation
   - `False`: Recompute full sequence each step (O(T²) complexity)
   - `True`: Cache K/V values, only compute new token (O(T) complexity)

2. **`kv_cache_quant`**: Whether to quantize KV-cache to INT8
   - `False`: Store K/V as full precision (float16/bfloat16)
   - `True`: Quantize to INT8 with per-channel scales (4x memory reduction)

3. **`cross_layer_sharing`**: Whether to share KV-cache across layer pairs
   - `False`: Each layer has its own K/V cache
   - `True`: Even layers compute K/V, odd layers reuse (2x cache reduction)

### Using the Version Registry

```python
from benchmark.versions import (
    VERSIONS,                    # Dict of all versions
    get_version_config,          # Get full config for a version
    load_model_for_version,      # Load model configured for version
    get_use_cache,               # Get use_cache flag
    get_kv_cache_quant,          # Get kv_cache_quant flag
    get_cross_layer_sharing,     # Get cross_layer_sharing flag
)

# Get version configuration
v_config = get_version_config('v2')
# Returns: {'description': 'KV-cache + INT8 quantization',
#           'use_cache': True, 'kv_cache_quant': True, 'cross_layer_sharing': False}

# Load model ready for benchmarking
model, v_config = load_model_for_version('v3', 'gpt2-medium', 'bfloat16')
# model is on CUDA, in eval mode, configured for v3
```

---

## Benchmarks

### 1. Latency vs T (`latency`)

**Purpose**: Measure total generation time and per-token latency across different generation lengths.

**Key Metrics**:
- `time_total_ms_median`: Median total time to generate T tokens
- `time_per_token_ms_median`: Median time per token (total / T)
- `time_total_ms_std`: Standard deviation for error bars

**What to Look For**:
- v0 shows O(T²) scaling (per-token time increases with T)
- v1-v4 show O(T) scaling (per-token time roughly constant)
- v2/v3/v4 may show slight overhead from quantization/sharing

```bash
python benchmark/main.py latency --version v0 v1 v2
```

### 2. VRAM vs T (`vram`)

**Purpose**: Measure peak GPU memory usage during generation.

**Key Metrics**:
- `peak_memory_bytes`: Maximum memory allocated during generation
- `peak_activation_bytes`: Memory above baseline (model weights)
- `baseline_memory_bytes`: Memory used by model weights alone

**What to Look For**:
- v0: Constant memory (no cache stored)
- v1: Linear memory growth with T (cache grows)
- v2: ~4x less cache memory than v1 (INT8 quantization)
- v3: ~2x less cache memory than v1 (layer sharing)
- v4: ~8x less cache memory than v1 (both optimizations)

```bash
python benchmark/main.py vram --version v1 v2 v3 v4
```

### 3. Per-Phase Timing (`phase`)

**Purpose**: Break down generation time by component.

**Key Metrics** (in `time_phase_ms`):
- `embedding`: Time in token + position embeddings
- `attention`: Time in self-attention layers
- `mlp`: Time in feed-forward layers
- `head`: Time in output projection
- `other`: Unaccounted time (sampling, concatenation)
- `total`: End-to-end time

**What to Look For**:
- Attention typically dominates (especially for v0)
- v1-v4 should show reduced attention time per step
- v3 may show reduced attention compute (borrower layers only compute Q)

```bash
python benchmark/main.py phase --version v0 v1
```

### 4. Perplexity (`perplexity.py`)

**Purpose**: Verify model quality is not degraded by optimizations.

**Key Metrics**:
- `perplexity`: exp(average loss) on WikiText-2
- `loss`: Average cross-entropy loss
- `degradation`: % change from baseline

**Important Note**: Perplexity uses forward pass only (teacher forcing), so all versions produce **identical** perplexity. This benchmark verifies model loading works correctly.

```bash
python benchmark/perplexity.py --version v0 v1 v2 v3 v4
```

### 5. Roofline Analysis (`roofline/`)

**Purpose**: Analyze computational efficiency (compute-bound vs memory-bound).

**Profiling Modes**:
- `v*_prefill`: Full forward pass on P tokens
- `v*_decode`: Single token decode with P-token cache

**Key Insight**: Decode steps are typically memory-bound (low arithmetic intensity) while prefill is compute-bound.

```bash
# Profile v2 decode step
python benchmark/roofline/profile_decode_step.py --version v2_decode --P 512 --model gpt2-medium
```

---

## Configuration

### config.yaml Reference

```yaml
# Model configuration (used if pretrained not set)
model_config:
  n_layer: 12
  n_head: 12
  n_embd: 768
  block_size: 1024
  vocab_size: 50257
  dropout: 0.0
  bias: true

# Pretrained model (overrides model_config)
pretrained: gpt2  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

# Data type
dtype: bfloat16  # float32, bfloat16, float16

# Generation parameters
prompt_length: 32      # Initial prompt length
batch_size: 1          # Batch size for generation

# Latency/VRAM benchmark parameters
T_values: [32, 64, 128, 256, 512]  # Generation lengths to test

# Per-phase timing parameters
T_star: 64  # Fixed T for phase breakdown

# Measurement parameters
num_warmup_runs: 3     # Warmup iterations (not measured)
num_measure_runs: 5    # Measured iterations

# Random seed for reproducibility
seed: 42

# Output file
log_file: benchmark/results.jsonl

# Versions to benchmark (can also set via CLI)
versions: [v0, v1]
```

### CLI Options

```bash
python benchmark/main.py [BENCHMARK...] [OPTIONS]

# Benchmarks: latency, vram, phase, all (default: all)

# Options:
--config PATH          # Config file (default: benchmark/config.yaml)
--preset MODEL         # Pretrained model: gpt2, gpt2-medium, gpt2-large, gpt2-xl
--version V [V ...]    # Versions to benchmark: v0, v1, v2, v3, v4
--clear-log            # Clear results file before running
```

---

## Usage Examples

### Basic Comparisons

```bash
# Compare baseline vs KV-cache
python benchmark/main.py --version v0 v1

# Compare all memory optimization versions
python benchmark/main.py vram --version v1 v2 v3 v4

# Full benchmark on larger model
python benchmark/main.py --preset gpt2-medium --version v0 v1 v2 v3 v4
```

### Research Workflows

```bash
# 1. Verify model quality first
python benchmark/perplexity.py --version v0 v1 v2 v3 v4 --preset gpt2

# 2. Run latency benchmark
python benchmark/main.py latency --version v0 v1 v2 v3 v4 --clear-log

# 3. Run memory benchmark
python benchmark/main.py vram --version v1 v2 v3 v4

# 4. Analyze phase breakdown
python benchmark/main.py phase --version v0 v1

# 5. Generate plots
python benchmark/plot_results.py
```

### Roofline Analysis Workflow

```bash
# Profile all phases for v1 vs v2
cd benchmark/roofline

# Using NCU (requires sudo)
sudo ./run_ncu.sh v1_prefill gpt2-medium 512
sudo ./run_ncu.sh v1_decode gpt2-medium 512
sudo ./run_ncu.sh v2_decode gpt2-medium 512

# Parse and plot
python parse_ncu_results.py
python plot_roofline.py
```

---

## Results & Analysis

### Output Format (JSONL)

Each line in `results.jsonl` is a JSON object:

```json
{
  "implementation_name": "baseline",
  "gpu_name": "NVIDIA RTX A6000",
  "gpu_total_vram": 51527024640,
  "python_version": "3.10.12",
  "pytorch_version": "2.1.0",
  "cuda_version": "12.1",
  "dtype": "bfloat16",
  "model_config": {
    "n_layer": 24,
    "n_head": 16,
    "n_embd": 1024,
    "block_size": 1024,
    "vocab_size": 50257,
    "kv_cache_quant": false,
    "cross_layer_sharing": false
  },
  "pretrained": "gpt2-medium",
  "prompt_length": 32,
  "batch_size": 1,
  "seed": 42,
  "benchmark_name": "latency_vs_T",
  "version": "v1",
  "version_description": "KV-cache enabled",
  "use_cache": true,
  "kv_cache_quant": false,
  "cross_layer_sharing": false,
  "T": 128,
  "time_total_ms_median": 245.32,
  "time_per_token_ms_median": 1.916
}
```

### Loading Results for Analysis

```python
import json
import pandas as pd

# Load all results
results = []
with open('benchmark/results.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

df = pd.DataFrame(results)

# Filter to latency benchmark
latency_df = df[df['benchmark_name'] == 'latency_vs_T']

# Compare v1 vs v2 speedup
v1 = latency_df[latency_df['version'] == 'v1']
v2 = latency_df[latency_df['version'] == 'v2']
speedup = v1['time_total_ms_median'].values / v2['time_total_ms_median'].values
```

---

## Extending the System

### Adding a New Benchmark Configuration

To create a custom benchmark configuration for specific experiments:

**1. Create a new config file** (e.g., `benchmark/config_memory_study.yaml`):

```yaml
# Custom config for memory optimization study
pretrained: gpt2-medium

dtype: bfloat16

# Focus on longer sequences to stress memory
prompt_length: 64
batch_size: 1
T_values: [128, 256, 512, 768, 1024]

# More measurements for statistical significance
num_warmup_runs: 5
num_measure_runs: 10

T_star: 256

seed: 42
log_file: benchmark/results_memory_study.jsonl

# Compare memory optimization versions
versions: [v1, v2, v3, v4]
```

**2. Run with your custom config**:

```bash
python benchmark/main.py vram --config benchmark/config_memory_study.yaml
```

**3. Common configuration patterns**:

```yaml
# Quick sanity check (fast iteration)
pretrained: gpt2
T_values: [32, 64]
num_warmup_runs: 1
num_measure_runs: 2

# Production benchmark (thorough)
pretrained: gpt2-xl
T_values: [32, 64, 128, 256, 512]
num_warmup_runs: 5
num_measure_runs: 20

# Long sequence study
prompt_length: 128
T_values: [256, 512, 768, 1024]

# Batch size scaling study
batch_size: 4
T_values: [64, 128, 256]
```

**4. Override config via CLI** (useful for quick experiments):

```bash
# Use custom config but override versions
python benchmark/main.py --config benchmark/config_memory_study.yaml --version v1 v4

# Use custom config but override model
python benchmark/main.py --config benchmark/config_memory_study.yaml --preset gpt2-xl
```

### Adding a New Version (v5, v6, etc.)

1. **Update `benchmark/versions.py`**:

```python
VERSIONS = {
    # ... existing versions ...
    'v5': {
        'description': 'KV-cache + my new optimization',
        'use_cache': True,
        'kv_cache_quant': False,
        'cross_layer_sharing': False,
        'my_new_flag': True,  # Add new flag
    },
}
```

2. **Update `model.py`**:
   - Add new config flag to `GPTConfig`
   - Implement the optimization in `CausalSelfAttention` and/or `GPT`

3. **Update `versions.py` helper functions**:

```python
def get_my_new_flag(version: str) -> bool:
    return get_version_config(version).get('my_new_flag', False)
```

4. **Update model loading** in `create_model()` if needed

5. **Add tests** in `tests/test_kv_cache.py`

### Adding a New Benchmark

1. **Create benchmark function in `main.py`**:

```python
def run_my_benchmark(config):
    """My new benchmark."""
    print("Running my benchmark...")

    versions = config.get('versions', ['v0', 'v1'])
    pretrained = config.get('pretrained', 'gpt2')

    for version in versions:
        model, v_config = load_model_for_version(version, pretrained, config['dtype'])
        model_config_dict = get_model_config_dict(model)

        # ... benchmark logic ...

        result = {
            'benchmark_name': 'my_benchmark',
            'version': version,
            # ... metrics ...
        }
        log_result(config, result, model_config_dict)

        del model
        torch.cuda.empty_cache()
```

2. **Add to main()**:

```python
if 'all' in args.benchmark or 'mybench' in args.benchmark:
    run_my_benchmark(config)
```

3. **Add plotting** in `plot_results.py`

### Adding a New Model Size

The system already supports all GPT-2 sizes via `--preset`:
- `gpt2` (124M params)
- `gpt2-medium` (350M params)
- `gpt2-large` (774M params)
- `gpt2-xl` (1.5B params)

To add a custom model size, update `model.py`:

```python
@classmethod
def from_pretrained(cls, model_type, ...):
    config_args = {
        # ... existing configs ...
        'my-custom-model': dict(n_layer=32, n_head=32, n_embd=2048),
    }[model_type]
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` or `T_values` in config
   - Use smaller model (`--preset gpt2`)
   - Ensure previous runs cleaned up (`torch.cuda.empty_cache()`)

2. **Inconsistent Results**
   - Set `seed` in config for reproducibility
   - Increase `num_warmup_runs` for GPU warmup
   - Increase `num_measure_runs` for statistical stability

3. **Version Not Found**
   - Check spelling (v0, v1, v2, v3, v4)
   - Run `python benchmark/versions.py` to list available versions

4. **Import Errors**
   - Run from project root directory
   - Ensure `benchmark/` is importable (has `__init__.py` if needed)

### Performance Tips

1. **Use bfloat16** for best balance of speed and accuracy
2. **Run benchmarks individually** to avoid memory fragmentation
3. **Clear log between experiments** (`--clear-log`)
4. **Profile with NCU** for deep hardware analysis
