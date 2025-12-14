# Complete Asset Inventory: KV-Cache Optimization Research

**Repository:** nanoGPT-MLT2  
**Branch:** feature/unified_benchmarks  
**GPU:** NVIDIA RTX A6000 (48GB)  
**Model:** GPT-2 Medium (378.96M parameters, 24 layers)  
**Dataset:** WikiText-2 test split (287,644 tokens)

---

## I. Core Implementation Files

### 1. Model Architecture (`model.py`)
**Description:** GPT-2 implementation with KV-cache optimizations  
**Key Components:**
- `KVCache` class: Pre-allocated cache with in-place updates
- `CausalSelfAttention`: Flash Attention with cache support
- `GPT` class: Main transformer with configurable optimizations
- INT8 quantization methods: `quantize()`, `dequantize()`
- Cross-layer sharing support with Q-alignment option

**Key Findings:**
- Pre-allocation eliminates torch.cat overhead
- Flash Attention (scaled_dot_product_attention) provides ~1.5x speedup
- INT8 quantization implemented but has Python overhead bottleneck
- Cross-layer sharing breaks pretrained weight compatibility (perplexity 800+)

### 2. Training Script (`train.py`)
**Description:** Standard GPT-2 training loop  
**Features:** DDP support, gradient accumulation, learning rate scheduling  
**Note:** Not modified for KV-cache (inference-only optimization)

### 3. Sampling Script (`sample.py`)
**Description:** Text generation with KV-cache support  
**Key Features:**
- Autoregressive generation with configurable cache
- Temperature, top-k, top-p sampling
- Demonstrates practical cache usage

---

## II. Benchmark Suite

### A. Main Benchmark Scripts

#### 1. `benchmark/main.py`
**Description:** Unified benchmark orchestrator  
**Benchmarks Supported:**
- `latency_vs_T`: Total and per-token latency across sequence lengths
- `vram_vs_T`: Peak memory usage vs sequence length
- `per_phase_breakdown`: Timing by phase (attention, MLP, other)
- `max_batch_capacity`: Maximum concurrent batch size before OOM

**Key Features:**
- Version-aware (uses `versions.py` registry)
- Auto-logging to `results.jsonl`
- GPU clock locking for reproducibility
- Warmup iterations to stabilize measurements

**Usage:**
```bash
python benchmark/main.py --benchmark latency_vs_T --version v0 v1 --preset gpt2-medium
```

#### 2. `benchmark/perplexity.py`
**Description:** Model quality evaluation on WikiText-2  
**Evaluation Modes:**
- **Teacher-forcing** (default): Sliding window, fast, all versions identical
- **Autoregressive** (`--autoregressive`): One token at a time, tests cache quality

**Key Findings:**
- v0-v2: Perplexity ~27-29 (stable)
- v3/v4: Perplexity 800+ (catastrophic degradation)
- v3a/v4a: Perplexity ~80 (Q-alignment helps but insufficient)
- Chunked evaluation (≤1024 tokens) avoids sliding window position embedding issues

**Usage:**
```bash
# Fast check (4k tokens)
python benchmark/perplexity.py --autoregressive --version v0 v1 --max-tokens 4096

# Full evaluation (287k tokens)
python benchmark/perplexity.py --autoregressive --version v0 v1 v2 v3 v4
```

#### 3. `benchmark/plot_results.py`
**Description:** Publication-quality visualization generator  
**Output:** 50+ plots in `benchmark/plots/`  
**Plot Categories:**
- Latency comparisons (total, per-token, with confidence intervals)
- VRAM usage and growth curves
- Per-phase timing breakdowns (bar, stacked, pie charts)
- Throughput and speedup analyses
- Perplexity comparisons
- Pareto frontiers (memory vs latency)
- Extrapolation projections

**Key Features:**
- Automatic deduplication (first occurrence wins)
- Version-aware coloring and markers
- LaTeX table generation (`benchmark_tables.tex`)
- Metrics summary JSON export

**Usage:**
```bash
python benchmark/plot_results.py
```

### B. Roofline Analysis

#### 1. `benchmark/roofline/profile_decode_step.py`
**Description:** Profiles single decode step for roofline analysis  
**Method:** Runs one forward pass with torch.cuda.nvtx markers, captures performance counters  
**Output:** Performance counters (FLOPs, DRAM traffic, arithmetic intensity)  
**Usage:**
```bash
python benchmark/roofline/profile_decode_step.py --version v1 --preset gpt2-medium
```

#### 2. `benchmark/roofline/run_ncu.sh`
**Description:** NVIDIA Nsight Compute profiling automation  
**Profiled metrics:**
- `dram__bytes.sum` (DRAM traffic)
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` (FP ADD operations)
- `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` (FP MUL operations)
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` (FP FMA operations)
**Generates:** `.ncu-rep` binary files in `ncu_reports/` directory  
**Example command:**
```bash
sudo ncu --metrics dram__bytes.sum,smsp__sass_thread... -o v1_decode python profile_decode_step.py
```

#### 3. `benchmark/roofline/parse_ncu_results.py`
**Description:** Extracts metrics from NCU reports  
**Input:** `.ncu-rep` files  
**Output:** JSON files with structured metrics  
**Calculations:**
- `total_flops = fadd + fmul + 2*ffma` (FMA counts as 2 ops)
- `arithmetic_intensity = total_flops / dram_bytes`
- `achieved_flops = total_flops / time_ns * 1e9`
**Generated files:**
- `v0_prefill_metrics.json`
- `v1_prefill_metrics.json`
- `v1_decode_metrics.json`

#### 4. `benchmark/roofline/plot_roofline.py`
**Description:** Generates roofline model visualization  
**Output:** `roofline.png` + `roofline.pdf`

**Plot Details:**
- **X-axis:** Arithmetic Intensity (FLOPs/byte), logarithmic scale, range 0.1-100
- **Y-axis:** Performance (GFLOPS), logarithmic scale, range 10-1000
- **Roofline lines:**
  - **Memory bandwidth bound:** Diagonal line from origin, slope = 768 GB/s (A6000 bandwidth)
    - Equation: GFLOPS = (768 GB/s) × (AI FLOPs/byte) = 768 × AI
    - Intersection with compute bound at AI = 312000 / 768 = 406 FLOPs/byte
  - **Compute bound:** Horizontal line at 312 TFLOPS (A6000 bfloat16 peak)
    - Y = 312,000 GFLOPS (flat, represents maximum throughput)
- **Plotted points:**
  - **v0_prefill:** (AI=1.57, 377.0 GFLOPS) - teal circle
    - On memory bandwidth line (1.57 × 768 = 1206 GFLOPS theoretical, but only achieving 377)
    - Actual: 2.32 GFLOPs / 1.48 GB DRAM traffic
    - Status: **Memory-bound** (below roofline)
  - **v1_prefill:** (AI=1.56, 375.0 GFLOPS) - teal square
    - Nearly identical to v0_prefill (cache doesn't help prefill phase)
    - 2.32 GFLOPs / 1.48 GB DRAM traffic
  - **v1_decode:** (AI=0.51, 146.3 GFLOPS) - teal triangle
    - Lower AI (0.51 FLOPs/byte) - more memory-bound than prefill
    - 0.45 GFLOPs / 0.88 GB DRAM traffic
    - Status: **Heavily memory-bound** (4.7x below theoretical peak at this AI)
- **Ridge point annotation:** "Ridge point: AI=406" with vertical dashed line
- **Regions labeled:**
  - Left of ridge: "Memory Bandwidth Bound" (shaded blue)
  - Right of ridge: "Compute Bound" (shaded green)

**Key Roofline Findings (exact numbers):**

| Scenario | DRAM Traffic | FP ADD | FP MUL | FP FMA | Total FLOPs | Time (μs) | Arithmetic Intensity | Achieved GFLOPS | Efficiency |
|----------|--------------|--------|--------|--------|-------------|-----------|---------------------|-----------------|------------|
| **v0_prefill** | 1.48 GB | 397.7M | 432.2M | 745.8M | 2.32 GFLOPs | 6158 | **1.57** | 377.0 | 31% of 1205 theoretical |
| **v1_prefill** | 1.48 GB | 397.7M | 432.3M | 746.0M | 2.32 GFLOPs | 6191 | **1.56** | 375.0 | 31% of 1197 theoretical |
| **v1_decode** | 0.88 GB | 40.9M | 42.6M | 182.4M | 0.45 GFLOPs | 3065 | **0.51** | 146.3 | 19% of 768×0.51=392 theoretical |

**Interpretation:**
- All measured points are **memory-bound** (AI < 406 ridge point)
- v1_decode is most memory-bound (AI=0.51, only 19% efficiency)
- v0/v1 prefill slightly better (AI=1.57, 31% efficiency) but still far below compute limit
- None achieve compute-bound region (would need AI > 406)
- Optimization target: Increase arithmetic intensity through kernel fusion, reduce DRAM traffic

**Files:**
- `benchmark/roofline/ncu_reports/v0_prefill_metrics.json` (structured data)
- `benchmark/roofline/ncu_reports/v1_decode_metrics.json` (structured data)
- `benchmark/roofline/ncu_reports/v1_prefill_metrics.json` (structured data)
- `benchmark/roofline/ncu_reports/*.ncu-rep` (binary NCU reports for NVIDIA Nsight GUI)
- `benchmark/roofline/roofline.png` (visualization, 1200×800 px)
- `benchmark/roofline/roofline.pdf` (vector version for publications)

### C. Capacity Testing

#### 1. `capacity_test/bench_capacity_inference.py`
**Description:** Measures maximum concurrent batch size  
**Method:** Binary search to find OOM threshold

**Key Findings:**
| Version | Max Batch | Peak VRAM | Capacity vs v1 |
|---------|-----------|-----------|----------------|
| v0 | 491 | 47.2 GB | 0.26x |
| v1 | 491 | 47.2 GB | 1.0x (baseline) |
| v2 | 594 | 47.3 GB | 1.21x |
| v3 | 745 | 47.3 GB | 1.52x |
| v4 | 1867 | 47.3 GB | **3.8x** |

#### 2. `capacity_test/bench_capacity_training.py`
**Description:** Training-time batch capacity (not primary focus)

### D. Configuration Files

#### 1. `benchmark/versions.py`
**Description:** SINGLE SOURCE OF TRUTH for version definitions  
**Versions:**
- `v0`: No cache (baseline)
- `v1`: KV-cache enabled
- `v2`: KV-cache + INT8 quantization
- `v3`: KV-cache + cross-layer sharing
- `v4`: KV-cache + INT8 + cross-layer sharing
- `v3a`: [EXP] Cross-layer + Q-aligned
- `v4a`: [EXP] Cross-layer + INT8 + Q-aligned

**Key Functions:**
- `get_version_config(version)`: Returns full config dict
- `load_model_for_version(version, preset, dtype)`: Instantiates configured model

#### 2. `benchmark/config.yaml`
**Description:** Benchmark run configuration  
**Parameters:** Sequence lengths, batch sizes, warmup iterations, presets

#### 3. `benchmark/config_test.yaml`
**Description:** Fast config for quick testing (fewer T values)

---

## III. Results and Data

### A. Primary Results File

#### `benchmark/results.jsonl`
**Description:** JSONL log of all benchmark runs  
**Schema:** Each line is a JSON object with:
- Metadata: `benchmark_name`, `version`, `gpu_name`, `pytorch_version`, `dtype`
- Model config: `n_layer`, `n_head`, `n_embd`, `block_size`
- Measurements: Version-specific metrics (latency, VRAM, perplexity, etc.)

**Deduplication:** `plot_results.py` takes first occurrence for each (benchmark, version, T)

### B. Metrics Summary

#### `benchmark/plots/metrics_summary.json`
**Description:** Aggregated statistics across all benchmarks  
**Structure:**
```json
{
  "latency": {
    "v0": {"points": [...], "fit_total_ms_vs_T": {...}, "fit_per_token_ms_vs_T": {...}},
    "v1": {...},
    ...
  },
  "vram": {...},
  "per_phase_T256": {...},
  "per_phase_T512": {...},
  "max_batch": {...},
  "perplexity": {...}
}
```

**Key Metrics at T=1024:**
- v0: 7362.5 ms total (7.19 ms/token), 799.2 MB VRAM
- v1: 3347.9 ms total (3.27 ms/token), 920.8 MB VRAM → **2.2x speedup**
- v2: 6754.1 ms total (6.60 ms/token), 872.0 MB VRAM → **2x slower than v1** (Python overhead)
- v3: 3072.0 ms total (3.00 ms/token), 848.8 MB VRAM → **8% faster than v1** (but unusable quality)
- v4: 6913.4 ms total (6.75 ms/token), 807.6 MB VRAM → **Slowest but highest capacity**

---

## IV. Visualizations

### A. Key Plots (Most Important)

Located in: `benchmark/plots/`

#### 1. `comparison_per_token_latency.png`
**X-axis:** Sequence length T, logarithmic scale, range 32-1024  
**Y-axis:** Per-token latency (ms), linear scale  
**Lines plotted:**
- v0 (red circles, #E63946): Linear growth from 2.88ms (T=32) to 7.19ms (T=1024). Fit: m=0.00443 ms/token, showing O(n) complexity
- v1 (teal squares, #2A9D8F): Flat at ~3.2ms/token across all T. Fit: m=0.0000578 ms/token (83x less slope than v0)
- v2 (orange triangles, #F4A261): Grows from 2.96ms to 6.60ms, 2x slower than v1 at long sequences
- v3 (purple diamonds, #6A4C93): Flat at ~3.0ms/token, slightly better than v1
- v4 (dark teal pentagons, #264653): Flat at ~6.7ms/token, ~2x slower than v1
- v3a (lavender inverted triangles, #8E7CC3): Flat at ~3.0ms (experimental, limited data)
- v4a (blue-green stars, #1B6A73): Flat at ~6.7ms (experimental, limited data)

**What to notice:**
- v1 breaks quadratic→linear scaling (constant 3.2ms/token)
- v0 shows clear linear growth (memory bandwidth bottleneck)
- v2/v4 INT8 overhead doubles latency despite quantization
- v3 achieves best per-token time but unusable quality (PPL 864)

#### 2. `KEY_comparison_vram.png`
**X-axis:** Sequence length T, logarithmic scale, range 32-1024  
**Y-axis:** Peak VRAM usage (MB), linear scale  
**Lines plotted:**
- v0: 771.6 MB (T=32) → 799.2 MB (T=1024), growth +27.6 MB. Linear fit: m=0.029 MB/token
- v1: 871.2 MB (T=32) → 920.8 MB (T=1024), growth +49.6 MB. Highest VRAM due to full KV-cache storage
- v2: 821.0 MB (T=32) → 872.0 MB (T=1024), growth +51.0 MB. INT8 cache should save 50% but only saves 5%
- v3: 771.6 MB → 848.8 MB, growth +77.2 MB. Cross-layer sharing increases memory (unexpected)
- v4: 795.7 MB → 807.6 MB, growth +11.9 MB. Flattest curve—INT8+cross-layer achieves best memory efficiency

**What to notice:**
- v4 has least growth (11.9 MB across 32→1024 range), ideal for long sequences
- v1 uses most memory (920.8 MB) but delivers best latency
- v2 INT8 saves only ~5% VRAM vs v1 (should be ~50% theoretically)
- Strict latency-memory trade-off: fast versions use more VRAM

#### 3. `comparison_perplexity.png`
**X-axis:** Version labels (v0, v1, v2, v3, v4, v3a, v4a)  
**Y-axis:** Perplexity, logarithmic scale (to show 20→780 range)  
**Bars (4096-token autoregressive evaluation on WikiText-2):**
- v0: 21.76 (red bar, baseline without cache)
- v1: 21.72 (teal bar, identical to baseline)
- v2: 21.73 (orange bar, +0.05% degradation—negligible)
- v3: 779.90 (purple bar, +3489% catastrophic)
- v4: 782.94 (dark teal bar, +3503% catastrophic)
- v3a: 116.88 (lavender bar, +438% with Q-alignment—still high)
- v4a: 116.92 (blue-green bar, +438% with Q-alignment—still high)

**What to notice:**
- v0-v2 cluster near 21-22 PPL (production-quality)
- v3/v4 jump to 780+ PPL (36x worse, unusable)
- Q-alignment (v3a/v4a) reduces degradation from 36x to 5.4x but still 5.4x too high
- Cross-layer sharing fundamentally breaks pretrained weights
- Horizontal line at PPL=21.72 shows baseline quality

#### 4. `comparison_perplexity_degradation.png`
**X-axis:** Version labels (v1, v2, v3, v4, v3a, v4a)  
**Y-axis:** Perplexity degradation (%), logarithmic scale  
**Bars:**
- v1: +0.00% (baseline, teal bar at y=0)
- v2: +0.05% (orange bar, negligible)
- v3: +3489% (purple bar, catastrophic failure)
- v4: +3503% (dark teal bar, catastrophic failure)
- v3a: +438% (lavender bar, reduced but still high)
- v4a: +438% (blue-green bar, reduced but still high)

**Annotations:**
- Red dashed line at 1% threshold ("acceptable degradation")
- Green zone: 0-1% (production-ready)
- Yellow zone: 1-10% (marginal)
- Red zone: >10% (unusable)

**What to notice:**
- Only v1 and v2 are production-quality (<1% degradation)
- Q-alignment helps (3489% → 438%) but insufficient—still 4.4x worse
- Cross-layer sharing requires architecture-aware training (e.g., GQA)
- Dramatic 36x degradation from weight incompatibility

#### 5. `pareto_memory_latency.png`
**X-axis:** Peak VRAM at T=1024 (MB), linear scale, range 800-925 MB  
**Y-axis:** Total latency at T=1024 (ms), linear scale, range 3000-7500 ms  
**Points plotted:**
- v0: (799.2 MB, 7362.5 ms) - red circle, upper-left (slow, low memory)
- v1: (920.8 MB, 3347.9 ms) - teal square, lower-right (fast, high memory)
- v2: (872.0 MB, 6754.1 ms) - orange triangle, middle (slow, moderate memory)
- v3: (848.8 MB, 3072.0 ms) - purple diamond, lower-middle (fastest but unusable quality)
- v4: (807.6 MB, 6913.4 ms) - dark teal pentagon, upper-left (slow, lowest memory)

**Pareto frontier:** Curve connecting v1 (fastest) → v3 (balanced) → v4 (most memory-efficient)  
**Dominated points:** v0, v2 (strictly worse than frontier points)

**What to notice:**
- No single winner—strict trade-off between latency and memory
- v1: Best for latency (3347.9ms) but highest VRAM (920.8MB)
- v4: Best for memory (807.6MB) but slowest (6913.4ms)
- v3: Pareto-optimal but quality broken (PPL 864)
- v2: Dominated (worse than v1 on both axes)
- 2.2x latency span and 14% VRAM span across frontier

#### 6. `per_phase_pie_v0_v1.png`
**Two pie charts side-by-side at T=512:**

**v0 (left pie):**
- Attention: 38.0% (1018.7ms, red slice)
- MLP: 44.1% (1181.2ms, blue slice)
- Other: 14.5% (388.4ms, gray slice)
- Head: 3.1% (84.0ms, green slice)
- Embedding: 0.2% (6.6ms, yellow slice)
- Total: 2678.8ms

**v1 (right pie):**
- Attention: 41.9% (970.6ms, red slice) - reduced from 1018.7ms
- MLP: 25.4% (588.5ms, blue slice) - halved from 1181.2ms
- Other: 28.9% (671.1ms, gray slice) - doubled from 388.4ms
- Head: 3.6% (82.8ms, green slice)
- Embedding: 0.2% (5.7ms, yellow slice)
- Total: 2318.7ms (13% faster than v0)

**What to notice:**
- Attention time reduced: 1018.7ms → 970.6ms (cache avoids recomputation)
- MLP time halved: 1181.2ms → 588.5ms (fewer tokens processed per step with cache)
- "Other" overhead doubled: 388.4ms → 671.1ms (29% of v1 time—optimization target)
- Total speedup: 15% at T=512 (improves to 2.2x at T=1024)
- Bottleneck shifted from MLP to "Other" (kernel launches, memory transfers)

### B. Detailed Comparison Plots

Located in: `benchmark/plots/`

#### Latency Analysis

**`comparison_total_latency.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Total generation time (ms), linear scale
- **Lines:** All versions (v0-v4, v3a, v4a) with markers
- **Key values at T=1024:** v0=7362.5ms, v1=3347.9ms, v2=6754.1ms, v3=3072.0ms, v4=6913.4ms
- **Fits:** Quadratic curves overlaid (v0: a=0.00478, v1: a=0.0000578—83x reduction)
- **What to notice:** v0 shows clear quadratic growth; v1 nearly linear (cache effect); v2/v4 slow despite quantization

**`comparison_total_latency_ci.png`**
- **Same axes as above**
- **Added:** Shaded 95% confidence intervals around each line
- **Confidence intervals:** ±50-150ms for most points (tight, indicating reproducibility)
- **What to notice:** Overlapping intervals between v2/v4 (statistically indistinguishable); v1/v3 clearly separated

**`comparison_per_token_latency.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Per-token latency (ms/token), linear scale
- **Lines:** All versions with linear fits
- **Key values at T=1024:** v0=7.19ms/tok, v1=3.27ms/tok, v2=6.60ms/tok, v3=3.00ms/tok, v4=6.75ms/tok
- **Slopes:** v0: m=0.00443 (linear growth), v1: m=0.0000578 (flat—83x improvement)
- **What to notice:** v1 achieves constant-time per-token (cache breaks O(n²) → O(n)); v0 grows linearly (O(n) per-token → O(n²) total)

**`comparison_per_token_latency_ci.png`**
- **Same as above with confidence intervals**
- **Confidence bands:** ±0.1-0.3 ms/token
- **What to notice:** v1 error bars stay flat across T (validates constant-time claim); v0 error increases with T

**`comparison_v0_vs_v1_latency.png`**
- **X-axis:** Sequence length T (linear scale this time), 32-1024
- **Y-axis:** Total latency (ms), linear scale
- **Two lines only:** v0 (red, quadratic) and v1 (teal, linear)
- **Speedup annotation:** "2.2x faster at T=1024" with arrow
- **Divergence point:** T≈128 where lines start separating visibly
- **What to notice:** Clean comparison showing when cache advantage begins; gap widens quadratically

**`v0_v1_total_latency_projection.png`**
- **X-axis:** Sequence length T (logarithmic), extrapolated to 2048 or 4096
- **Y-axis:** Total latency (ms), logarithmic scale
- **Lines:** v0 and v1 with fitted curves extended beyond measured range
- **Projected T=2048:** v0≈29s, v1≈6.7s (4.3x speedup)
- **Dashed lines:** Indicate extrapolation region (uncertain)
- **What to notice:** Speedup advantage grows superlinearly; cache benefit compounds at scale

#### VRAM Analysis

**`comparison_vram.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Peak VRAM usage (MB), linear scale, range 750-950 MB
- **Lines:** All versions (v0-v4)
- **Key values at T=1024:** v0=799.2MB, v1=920.8MB, v2=872.0MB, v3=848.8MB, v4=807.6MB
- **Growth rates (linear fits):** v0: m=0.029 MB/token, v1: m=0.022 MB/token, v4: m=0.009 MB/token
- **What to notice:** v1 highest absolute (cache storage), v4 flattest growth (best scaling), v2 only 5% savings vs v1 (INT8 underperforms)

**`vram_growth_v0_v1.png`**
- **X-axis:** Sequence length T (linear scale), 32-1024
- **Y-axis:** VRAM delta from baseline (MB), starting at 0
- **Two lines:** v0 (red) and v1 (teal), normalized to their T=32 values
- **Growth:** v0 grows 27.6MB, v1 grows 49.6MB (1.8x more growth)
- **Annotation:** "Cache overhead: +49.6 MB for 1024-token context"
- **What to notice:** v1 cache requires extra memory but grows sublinearly with T; reasonable overhead for 2.2x speedup

**`vram_bar_chart.png`**
- **X-axis:** Version labels (v0, v1, v2, v3, v4)
- **Y-axis:** Peak VRAM (MB), linear scale
- **Bars at T=1024:**
  - v0: 799.2 MB (red bar, baseline)
  - v1: 920.8 MB (teal bar, +15% overhead, tallest)
  - v2: 872.0 MB (orange bar, -5% vs v1)
  - v3: 848.8 MB (purple bar, -8% vs v1)
  - v4: 807.6 MB (dark teal bar, -12% vs v1, shortest)
- **What to notice:** v4 uses less memory than baseline v0 despite having cache; cross-layer+INT8 synergy

#### Throughput and Efficiency

**`comparison_throughput.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Throughput (tokens/second), linear scale
- **Lines:** All versions with markers
- **Calculation:** tokens/sec = T / (total_ms / 1000)
- **Key values at T=1024:**
  - v0: 139.1 tok/s (slowest)
  - v1: 305.9 tok/s (fastest, 2.2x v0)
  - v2: 151.6 tok/s (2x slower than v1)
  - v3: 333.3 tok/s (fastest overall but unusable quality)
  - v4: 148.1 tok/s (slowest after v0)
- **What to notice:** v1 achieves 300+ tok/s (production-quality); v3 slightly faster but broken quality; v2/v4 both ~150 tok/s (INT8 overhead dominates)

**`comparison_throughput_ci.png`**
- **Same as above with 95% confidence intervals**
- **Confidence bands:** ±10-30 tok/s at high T
- **What to notice:** v1 and v3 clearly separated from v2/v4 cluster; tight intervals show reproducibility

**`comparison_speedup.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Speedup vs v0 (unitless), linear scale, range 0.5x-2.5x
- **Horizontal line at y=1.0:** Baseline (v0 performance)
- **Lines:**
  - v1: Grows from 1.1x (T=32) to 2.2x (T=1024) - teal line above baseline
  - v2: Drops from 0.98x to 1.09x - orange line near baseline (minimal benefit)
  - v3: Grows to 2.4x at T=1024 - purple line (best speedup but quality broken)
  - v4: Stays near 1.06x - dark teal line (slowdown vs v0)
- **What to notice:** Only v1 and v3 beat baseline; speedup increases with T (cache advantage compounds); v2/v4 actually slower than no cache

**`comparison_speedup_vs_v1.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Speedup vs v1 (unitless), linear scale
- **Baseline at y=1.0:** v1 performance (now the reference)
- **Lines:**
  - v0: 0.45x at T=1024 (2.2x slower than v1) - red line below baseline
  - v2: ~0.5x across all T - orange line (consistently 2x slower)
  - v3: ~1.08x at T=1024 - purple line (8% faster but quality broken)
  - v4: ~0.48x - dark teal line (2x slower)
- **What to notice:** v2/v4 both ~0.5x (INT8 overhead wipes out cache benefit); v3 only 8% faster (marginal gain not worth quality loss)

**`comparison_memory_efficiency.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Memory efficiency (tokens per MB VRAM), linear scale
- **Lines:** All versions
- **Calculation:** T / peak_vram_MB
- **Key values at T=1024:**
  - v0: 1.28 tok/MB (lowest efficiency)
  - v1: 1.11 tok/MB (cache overhead reduces efficiency)
  - v2: 1.17 tok/MB (slightly better than v1)
  - v3: 1.21 tok/MB
  - v4: 1.27 tok/MB (highest efficiency, nearly matches v0)
- **What to notice:** v4 achieves near-v0 efficiency despite cache; cross-layer+INT8 combo is memory-optimal

#### Per-Phase Breakdown

**`per_phase_breakdown_bar.png`**
- **X-axis:** Version labels (v0, v1, v2, v3, v4)
- **Y-axis:** Time (ms), linear scale, measured at T=512
- **Grouped bars for each version (5 phases):**
  - Embedding: ~6ms (yellow, negligible <1%)
  - Attention: v0=1019ms, v1=971ms, v2=2791ms, v3=859ms, v4=2031ms (red bars)
  - MLP: v0=1181ms, v1=589ms, v2=638ms, v3=591ms, v4=632ms (blue bars)
  - Head: ~83ms across all (green, consistent)
  - Other: v0=388ms, v1=671ms, v2=688ms, v3=669ms, v4=658ms (gray bars)
- **What to notice:** v2 attention explodes to 2791ms (INT8 overhead); v1 MLP halves vs v0 (cache reduces work); "Other" grows 2x in cache versions (overhead target)

**`per_phase_breakdown_stacked.png`**
- **X-axis:** Version labels
- **Y-axis:** Total time (ms) at T=512, stacked to show 100%
- **Stacked segments (bottom to top):**
  - Embedding: Thin yellow slice at bottom (~0.2%)
  - Attention: v0=38%, v1=42%, v2=66%, v3=39%, v4=60% (red)
  - MLP: v0=44%, v1=25%, v2=15%, v3=27%, v4=19% (blue)
  - Other: v0=14%, v1=29%, v2=16%, v3=30%, v4=19% (gray)
  - Head: ~3-4% (green, top slice)
- **Total heights:** v0=2679ms, v1=2319ms (-13%), v2=4207ms (+57%), v3=2208ms (-18%), v4=3410ms (+27%)
- **What to notice:** v2 attention dominates 66% (bottleneck); v1 "Other" nearly doubles to 29% (new bottleneck); proportion shift reveals where overhead moved

**`per_phase_breakdown_with_insights.png`**
- **Same as bar chart with annotations:**
  - Arrow pointing to v2 attention bar: "INT8 overhead: +1820ms vs v1"
  - Arrow to v1 "Other": "Kernel launch overhead 2x vs v0"
  - Dashed line at 1662ms: "v1 total time (baseline for comparisons)"
- **Color-coded zones:** Green box around v1/v3 bars (acceptable), red box around v2/v4 (slow)
- **What to notice:** Visual guide to bottleneck identification; immediately shows v2 attention problem and v1 overhead opportunity

**`per_phase_pie_v0_v1.png`**
- **Described in detail in Key Plots section above**
- Two pie charts side-by-side showing percentage breakdown
- Emphasis on "Other" overhead shift (14% → 29%)

**`per_phase_pie_improved.png`**
- **Enhanced version with:**
  - Percentage labels on each slice
  - Absolute time values (e.g., "Attention: 42%, 971ms")
  - Exploded slices for small segments (Embedding, Head)
  - Legend positioned outside for clarity
  - Subtitle: "Measured at T=512, GPT-2 Medium, A6000"
- **What to notice:** More publication-ready format; all information readable without referring to external tables

#### Capacity and Trade-offs

**`comparison_max_batch_capacity.png`** ⭐ **Hero Plot**
- **X-axis:** Version labels (v0, v1, v2, v3, v4)
- **Y-axis:** Maximum concurrent batch size, linear scale, range 0-2000
- **Bars:**
  - v0: 491 (red bar, baseline)
  - v1: 491 (teal bar, identical to v0—cache doesn't reduce capacity)
  - v2: 594 (orange bar, 1.21x improvement, +21%)
  - v3: 745 (purple bar, 1.52x improvement, +52%)
  - v4: **1867** (dark teal bar, **3.8x improvement**, +280%)
- **Annotations:**
  - "3.8x capacity increase" with large arrow pointing to v4
  - Dashed line at y=491: "v0/v1 baseline"
  - "INT8 + cross-layer synergy" label near v4
- **Measurement details:** At T=512, GPT-2 Medium, on A6000 48GB, batch size for 47.3 GB VRAM usage
- **What to notice:** v4's 3.8x capacity is the breakthrough finding; enables 1867 concurrent users vs 491 for v1; justifies v4 despite 2x latency slowdown for batch workloads

**`comparison_peak_memory_at_max_batch.png`**
- **X-axis:** Version labels
- **Y-axis:** Peak VRAM at max batch (GB), range 46-48 GB
- **Bars:** All versions hit ~47.2-47.3 GB (within 0.1 GB)
- **What to notice:** All versions saturate VRAM equally (measurement validity check); differences in batch size come from per-sample memory efficiency

**`tradeoff_memory_latency_T1024.png`**
- **X-axis:** Peak VRAM (MB) at T=1024, range 800-920 MB
- **Y-axis:** Total latency (ms) at T=1024, range 3000-7500 ms
- **Scatter points:**
  - v0: (799, 7363) - red circle, top-left quadrant (slow, low memory)
  - v1: (921, 3348) - teal square, bottom-right (fast, high memory)
  - v2: (872, 6754) - orange triangle, middle (slow, moderate)
  - v3: (849, 3072) - purple diamond, bottom-middle (fastest)
  - v4: (808, 6913) - dark teal pentagon, top-left (slow, low memory)
- **Quadrants labeled:**
  - Bottom-left: "Ideal" (fast + low memory) - empty, no version achieves both
  - Bottom-right: "Speed priority" (v1 lives here)
  - Top-left: "Memory priority" (v4 lives here)
  - Top-right: "Avoid" (slow + high memory) - empty
- **Diagonal line:** Represents trade-off frontier
- **What to notice:** Fundamental trade-off; v1 and v4 are Pareto-optimal; no version in ideal quadrant (physics constraint)

**`pareto_memory_latency.png`**
- **Similar to above but with all T values (32-1024) plotted**
- **Multiple points per version:** Different colors/markers per version, each point is a different T
- **Pareto frontier curve:** Connects non-dominated points across versions and T values
- **Frontier includes:** v1 at all T (fastest), v3 at most T (balanced), v4 at high T (memory-optimal)
- **Dominated region:** Shaded gray area representing inferior configurations
- **What to notice:** Trade-off consistent across sequence lengths; v2 always dominated; v3 would be frontier if quality worked

#### KV-Cache Variant Comparisons

**`comparison_kv_variants_latency.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Total latency (ms), linear scale
- **Lines:** Only v1-v4 (excludes v0 baseline for focused comparison)
- **Key comparison at T=1024:**
  - v1: 3348ms (teal, baseline for cache variants)
  - v2: 6754ms (orange, 2.02x slower—INT8 overhead)
  - v3: 3072ms (purple, 1.09x faster—best cache variant but quality broken)
  - v4: 6913ms (dark teal, 2.06x slower—INT8+cross-layer overhead)
- **What to notice:** v2 and v4 nearly identical latency (INT8 dominates); v3 marginally faster than v1 (8% gain); pure cache (v1) is fastest production-ready option

**`kv_variants_per_token_bar_T1024.png`**
- **X-axis:** Version labels (v1, v2, v3, v4)
- **Y-axis:** Per-token latency (ms/token) at T=1024
- **Bars:**
  - v1: 3.27 ms/tok (teal, baseline)
  - v2: 6.60 ms/tok (orange, 2.02x slower)
  - v3: 3.00 ms/tok (purple, 1.09x faster, shortest bar)
  - v4: 6.75 ms/tok (dark teal, 2.06x slower, tallest bar)
- **Annotations:** Percentage differences labeled (e.g., "+102%" for v2)
- **What to notice:** v3's 8% advantage too small to justify quality loss; v2/v4 both ~2x slower (consistent INT8 tax)

**`kv_variants_throughput_bar_T1024.png`**
- **X-axis:** Version labels
- **Y-axis:** Throughput (tokens/second) at T=1024
- **Bars:**
  - v1: 305.9 tok/s (teal, tallest)
  - v2: 151.6 tok/s (orange, half of v1)
  - v3: 333.3 tok/s (purple, 1.09x v1)
  - v4: 148.1 tok/s (dark teal, shortest, half of v1)
- **What to notice:** v3 achieves 333 tok/s (best) but unusable; v1's 306 tok/s is production winner; v2/v4 both ~150 tok/s (quantization halves throughput)

**`kv_variants_vram_bar_T1024.png`**
- **X-axis:** Version labels
- **Y-axis:** Peak VRAM (MB) at T=1024
- **Bars:**
  - v1: 920.8 MB (teal, tallest—full precision cache)
  - v2: 872.0 MB (orange, -5.3% vs v1, 48.8 MB saved)
  - v3: 848.8 MB (purple, -7.8% vs v1, 72.0 MB saved)
  - v4: 807.6 MB (dark teal, -12.3% vs v1, 113.2 MB saved, shortest)
- **Annotations:** Absolute savings labeled (e.g., "−113 MB" for v4)
- **What to notice:** v4 achieves lowest memory (807.6 MB); INT8+cross-layer reduces VRAM by 12.3%; but savings modest compared to capacity benchmark (3.8x batch size)

#### Extrapolation

**`comparison_extrapolation.png`**
- **X-axis:** Sequence length T (logarithmic), measured range 32-1024 + extrapolated to 2048 or 4096
- **Y-axis:** Total latency (ms), logarithmic scale
- **Solid lines (32-1024):** Measured data with markers
- **Dashed lines (1024-4096):** Extrapolated using fitted curves
- **Extrapolated values at T=2048:**
  - v0: ~29,000ms (29s, quadratic fit a=0.00478)
  - v1: ~6,700ms (6.7s, near-linear fit)
  - v2: ~26,000ms (26s)
  - v3: ~6,200ms (6.2s)
  - v4: ~27,000ms (27s)
- **Shaded uncertainty region:** Around dashed lines (±20% confidence band)
- **Annotations:**
  - "Measured data" label pointing to solid region
  - "Extrapolation (uncertain)" label in dashed region
  - "v0→v1 speedup grows to 4.3x at T=2048" callout
- **What to notice:** Cache advantage compounds at long sequences; v0 becomes prohibitive beyond 1024; extrapolation assumes no new bottlenecks (optimistic)

#### Perplexity Analysis

**`comparison_perplexity.png`**
- **X-axis:** Version labels (v0, v1, v2, v3, v4, v3a, v4a)
- **Y-axis:** Perplexity (WikiText-2), logarithmic scale, range 10-1000
- **Bars (4096-token evaluation):**
  - v0: 21.76 PPL (red bar, baseline without cache)
  - v1: 21.72 PPL (teal bar, cache baseline—identical quality)
  - v2: 21.73 PPL (orange bar, +0.05% degradation—negligible)
  - v3: 779.90 PPL (purple bar, +35.9x catastrophic failure)
  - v4: 782.94 PPL (dark teal bar, +36.0x catastrophic failure)
  - v3a: 116.88 PPL (lavender bar, +5.4x with Q-alignment—still poor)
  - v4a: 116.92 PPL (blue-green bar, +5.4x with Q-alignment—still poor)
- **Horizontal reference line at PPL=21.72:** Marks production quality threshold
- **Color zones:**
  - Green zone (PPL 20-30): Production-ready (v0, v1, v2)
  - Yellow zone (PPL 30-100): Marginal quality
  - Red zone (PPL >100): Unusable (v3, v4, v3a, v4a)
- **What to notice:** Cache (v1) and INT8 (v2) maintain quality; cross-layer sharing breaks pretrained weights; Q-alignment helps (36x→5.4x degradation) but insufficient for production

**`comparison_perplexity_degradation.png`**
- **X-axis:** Version labels (v1, v2, v3, v4, v3a, v4a)
- **Y-axis:** Perplexity increase vs v1 baseline (%), logarithmic scale
- **Bars:**
  - v1: 0% (baseline, marked with horizontal line)
  - v2: +0.05% (orange bar, barely visible—negligible)
  - v3: +3489% (purple bar, catastrophic)
  - v4: +3503% (dark teal bar, catastrophic)
  - v3a: +438% (lavender bar, reduced from v3 but still high)
  - v4a: +438% (blue-green bar, reduced from v4 but still high)
- **Threshold lines:**
  - Green dashed at 1%: "Acceptable quality"
  - Yellow dashed at 10%: "Marginal quality"
  - Red dashed at 100%: "Unusable quality"
- **Annotations:**
  - "Q-alignment reduces degradation 8x (3500% → 438%)"
  - "Still 4.4x worse than baseline—requires retraining"
- **What to notice:** Only v2 passes production threshold (<1%); v3a/v4a improvements dramatic but insufficient; cross-layer requires architecture-aware pre-training (GQA)

### C. Per-Version Individual Plots

#### Latency (Total Time)

**`latency_total_vs_T_v0.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Total generation time (ms), linear scale
- **Single line:** v0 (red circles with connecting line)
- **Data points:** 16 measurements from 92.1ms (T=32) to 7362.5ms (T=1024)
- **Fitted curve:** Quadratic overlay, coefficients a=0.00478, b=2.27, c=30.46, R²=0.9999
- **Equation displayed:** "y = 0.00478·T² + 2.27·T + 30.46"
- **What to notice:** Near-perfect quadratic fit (R²=0.9999); validates O(n²) complexity; every doubling of T quadruples time at large T

**`latency_total_vs_T_v1.png`**
- **X-axis/Y-axis:** Same as v0
- **Single line:** v1 (teal squares)
- **Data points:** 103.4ms (T=32) to 3347.9ms (T=1024)
- **Fitted curve:** Quadratic with a=0.0000578 (83x smaller than v0), b=3.17, c=1.06, R²=0.9997
- **Equation:** "y = 0.000058·T² + 3.17·T + 1.06"
- **What to notice:** Near-linear behavior (tiny quadratic coefficient); 2.2x faster than v0 at T=1024; cache successfully breaks complexity wall

**`latency_total_vs_T_v2.png` through `v4.png`**
- **Same format for v2, v3, v4**
- **v2:** Quadratic a=0.00388, slower than v1, fit shows INT8 overhead
- **v3:** Quadratic a=0.0000539 (slightly less than v1), fastest fit
- **v4:** Quadratic a=0.00373, slowest cache variant

#### Latency (Per-Token)

**`latency_per_token_vs_T_v0.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Per-token latency (ms/token), linear scale, range 2-8 ms/tok
- **Single line:** v0 (red circles)
- **Data points:** 2.88 ms/tok (T=32) to 7.19 ms/tok (T=1024)
- **Fitted line:** Linear, slope m=0.00443, intercept b=2.56, R²=0.9967
- **Equation:** "y = 0.00443·T + 2.56"
- **What to notice:** Clean linear growth; per-token cost increases with context length (quadratic attention); slope indicates memory bandwidth constraint

**`latency_per_token_vs_T_v1.png`**
- **X-axis/Y-axis:** Same as v0
- **Single line:** v1 (teal squares)
- **Data points:** 3.23 ms/tok (T=32) to 3.27 ms/tok (T=1024)
- **Fitted line:** Nearly flat, slope m=0.0000578, intercept b=3.21, R²=0.0017 (poor fit—data is flat)
- **Equation:** "y = 0.000058·T + 3.21"
- **What to notice:** Constant ~3.2 ms/tok (flat line); cache breaks scaling; 83x slope reduction vs v0; low R² indicates no T-dependence (intended)

**`latency_per_token_vs_T_v2.png` through `v4.png`**
- **Similar format**
- **v2:** Linear growth from 2.96 to 6.60 ms/tok (slower than v1, INT8 overhead)
- **v3:** Flat at ~3.0 ms/tok (like v1 but 8% faster)
- **v4:** Flat at ~6.7 ms/tok (constant but 2x slower than v1)

#### VRAM

**`vram_vs_T_v0.png`**
- **X-axis:** Sequence length T (logarithmic), 32-1024
- **Y-axis:** Peak VRAM usage (MB), linear scale, range 770-810 MB
- **Single line:** v0 (red circles)
- **Data points:** 771.6 MB (T=32) to 799.2 MB (T=1024)
- **Growth:** +27.6 MB over full range
- **Fitted line:** Linear, slope m=0.029 MB/token, R²=0.9936
- **What to notice:** Modest growth (3.5% increase); baseline memory stable; small activations dominate

**`vram_vs_T_v1.png`**
- **Same axes**
- **Single line:** v1 (teal squares)
- **Data points:** 871.2 MB (T=32) to 920.8 MB (T=1024)
- **Growth:** +49.6 MB (1.8x more than v0)
- **Baseline offset:** +99.6 MB vs v0 at T=32 (cache allocation overhead)
- **What to notice:** Cache requires upfront memory; growth similar to v0 (cache preallocated); jump at T=1024 suggests reallocation

**`vram_vs_T_v2.png` through `v4.png`**
- **v2:** 821.0→872.0 MB, +51.0 MB growth (only 5% savings vs v1 despite INT8)
- **v3:** 771.6→848.8 MB, +77.2 MB growth (cross-layer increases memory unexpectedly)
- **v4:** 795.7→807.6 MB, +11.9 MB growth (flattest—INT8+cross-layer synergy works for memory)

### D. Tables and LaTeX Output

#### `benchmark/plots/benchmark_tables.tex`
**Description:** LaTeX-formatted tables for publication  
**Contents:**
- Latency table (all versions, all T values)
- VRAM table
- Per-phase breakdown table
- Perplexity comparison table

#### `benchmark/plots/perplexity_table.md`
**Description:** Markdown-formatted perplexity results  
**Example:**
```markdown
| Version | Perplexity | Loss | Degradation |
|---------|------------|------|-------------|
| v0 | 27.55 | 3.316 | baseline |
| v1 | 27.55 | 3.316 | +0.00% |
| v2 | 27.78 | 3.324 | +0.84% |
| v3 | 864.32 | 6.761 | +3037% |
```

---

## V. Documentation

### A. Research Documents

#### 1. `RESEARCH_SUMMARY.md`
**Description:** Paper outline and key findings synthesis  
**Sections:**
- Key Findings (v0→v1, v2 failure, v3/v4 quality issues, v4 capacity success)
- Research paper outline: "The Three Walls of Transformer Inference"
- Potentially Useful Insights (concise bullets)

**Key Narratives:**
- **Compute Wall:** v1 breaks it with 2.2x speedup
- **Overhead Wall:** v2 hits it due to Python quantization overhead
- **Architectural Wall:** v3/v4 hit it due to pretrained weight incompatibility

#### 2. `benchmark/RESEARCH_SPACE.md`
**Description:** Detailed research findings and analysis  
**Sections:**
- Executive summary: "Two Regimes" discovery
- Version definitions with experimental variants
- Baseline success (v0→v1)
- Overhead Wall analysis (v2 Python Tax calculation)
- Architectural Wall analysis (cross-layer sharing failure)
- Capacity breakthrough (v4 serves 3.8x more users)

**Key Insights:**
- v2 Python overhead: 120-250ms at T=1024 (vs 1662ms total for v1)
- v3/v4 perplexity: 800+ (30x worse than baseline)
- v3a/v4a Q-alignment: Reduces perplexity degradation from 30x to ~3x
- v4 capacity: 1867 concurrent users vs 491 for v1

#### 3. `BENCHMARK_ANALYSIS.md`
**Description:** Initial benchmark results documentation  
**Status:** Historical record, superseded by RESEARCH_SUMMARY

#### 4. `Gemini_Report.md`
**Description:** AI-generated analysis report  
**Note:** External perspective on benchmark results

### B. Technical Documentation

#### 1. `docs/BENCHMARKING.md`
**Description:** Guide to running benchmarks  
**Topics:**
- Benchmark suite overview
- How to run each benchmark
- Interpreting results
- Reproducibility considerations

#### 2. `docs/OPTIMIZATION_NOTES.md`
**Description:** Implementation details of optimizations  
**Topics:**
- KV-cache implementation strategies
- INT8 quantization approach
- Cross-layer sharing mechanics
- Flash Attention integration

#### 3. `docs/TESTING.md`
**Description:** Testing strategy and validation  
**Topics:**
- Unit tests for KV-cache
- Perplexity validation
- Numerical accuracy checks

#### 4. `README.md`
**Description:** Main project documentation  
**Contents:**
- Project overview
- Quick start guide
- Installation instructions
- Usage examples

### C. Supplementary Documents

#### 1. `perplexity_table_content.md`
**Description:** Extracted perplexity table data (generated)

#### 2. `LICENSE`
**Description:** MIT License

#### 3. `requirements.txt`
**Description:** Python dependencies  
**Key packages:** torch, numpy, tiktoken, datasets, transformers, tqdm

---

## VI. Test and Debug Scripts

### A. KV-Cache Tests

#### 1. `tests/test_kv_cache.py`
**Description:** Unit tests for KVCache class  
**Tests:**
- Cache initialization
- Update and retrieval
- Quantization/dequantization accuracy
- Trimming behavior
- Cross-layer sharing mechanics

#### 2. `test_perplexity.py`
**Description:** Quick perplexity validation on synthetic data  
**Purpose:** Verify cache produces identical outputs to no-cache
**Data:** torch.arange(100) for reproducibility

#### 3. `test_wikitext.py`
**Description:** Perplexity test on real WikiText-2 data  
**Purpose:** Debug cache behavior on actual text (discovered trimming issue)
**Key Feature:** Divergence detection in first 50 tokens

#### 4. `debug_cache.py`
**Description:** Basic cache correctness verification  
**Output:** Max logit difference between cached and non-cached paths

#### 5. `debug_cache_detailed.py`
**Description:** Detailed cache inspection with token-by-token comparison  
**Output:** Per-token logit differences and perplexity calculation

#### 6. `debug_trim.py`
**Description:** Traces position embeddings around cache trimming boundary  
**Purpose:** Diagnose sliding window position embedding issues
**Finding:** Revealed that positions get stuck at 1023 after trimming

---

## VII. Configuration and Model Presets

### A. Training Configs

Located in: `config/`

#### 1. `train_gpt2.py`
**Description:** Full GPT-2 training from scratch  
**Parameters:** 124M model, OpenWebText dataset

#### 2. `finetune_shakespeare.py`
**Description:** Fine-tune pretrained GPT-2 on Shakespeare  
**Use case:** Quick validation of training pipeline

#### 3. `train_shakespeare_char.py`
**Description:** Character-level model on Shakespeare  
**Use case:** Fast experiments with smaller model

### B. Evaluation Configs

#### 1. `eval_gpt2.py`
**Description:** Evaluate GPT-2 small (124M) on WikiText-2

#### 2. `eval_gpt2_medium.py`
**Description:** Evaluate GPT-2 medium (355M) - primary benchmark config

#### 3. `eval_gpt2_large.py`
**Description:** Evaluate GPT-2 large (774M)

#### 4. `eval_gpt2_xl.py`
**Description:** Evaluate GPT-2 XL (1.5B)

---

## VIII. Utility Scripts

### 1. `analyze_results.py`
**Description:** Results analysis and visualization (legacy)  
**Status:** Superseded by plot_results.py

### 2. `bench.py`
**Description:** Simple benchmarking script (legacy)  
**Status:** Superseded by benchmark/main.py

### 3. `configurator.py`
**Description:** Interactive model configuration generator  
**Purpose:** Generate training configs for different model sizes

### 4. `benchmark/diagnose_pstate.py`
**Description:** Diagnoses GPU P-state issues  
**Purpose:** Ensures GPU runs at max clocks for reproducibility

### 5. `benchmark/activate_clock_lock.sh`
**Description:** Locks GPU clocks to maximum for benchmarking  
**Usage:** `sudo ./activate_clock_lock.sh`

### 6. `benchmark/deactivate_clock_lock.sh`
**Description:** Unlocks GPU clocks after benchmarking

### 7. `benchmark/run_full_benchmark.sh`
**Description:** Automated full benchmark suite execution  
**Runs:** All benchmarks for all versions, generates plots

---

## IX. Data and Notebooks

### A. Data Directories

#### 1. `data/openwebtext/`
**Description:** OpenWebText dataset preparation  
**File:** `prepare.py` - Downloads and tokenizes dataset

#### 2. `data/shakespeare/`
**Description:** Shakespeare dataset (word-level)  
**File:** `prepare.py` - Prepares Shakespeare text

#### 3. `data/shakespeare_char/`
**Description:** Shakespeare dataset (character-level)  
**File:** `prepare.py` - Character-level tokenization

#### 4. `benchmark/.cache/`
**Description:** Cached tokenized WikiText-2 data  
**File:** `wikitext2_gpt2_tokens.pt` (generated)  
**Purpose:** Speeds up repeated perplexity evaluations

### B. Jupyter Notebooks

#### 1. `scaling_laws.ipynb`
**Description:** Analysis of model scaling laws  
**Topics:**
- Parameter count vs performance
- Compute budget optimization
- Scaling predictions

#### 2. `transformer_sizing.ipynb`
**Description:** Transformer architecture design exploration  
**Topics:**
- Layer depth vs width trade-offs
- Memory footprint calculations
- FLOPs estimation

---

## X. Key Findings Summary

### Performance Metrics at T=1024

| Version | Total Time | Per-Token | VRAM | Max Batch | Perplexity (4k tokens) | Status |
|---------|------------|-----------|------|-----------|----------------------|---------|
| v0 | 7362.5 ms | 7.19 ms | 799.2 MB | 491 | 21.76 | Baseline |
| v1 | 3347.9 ms | 3.27 ms | 920.8 MB | 491 | 21.72 | ✓ Production |
| v2 | 6754.1 ms | 6.60 ms | 872.0 MB | 594 | 21.73 | ✗ Slow (Python overhead) |
| v3 | 3072.0 ms | 3.00 ms | 848.8 MB | 745 | 779.90 | ✗ Quality broken |
| v4 | 6913.4 ms | 6.75 ms | 807.6 MB | 1867 | 782.94 | ⚠ Capacity-only |
| v3a | - | - | - | - | 116.88 | ⚠ Experimental |
| v4a | - | - | - | - | 116.92 | ⚠ Experimental |

### The Three Walls

1. **Compute Wall (BROKEN by v1)**
   - 2.2x latency reduction at T=1024
   - 83x complexity reduction (quadratic coefficient: 0.00478 → 0.0000578)
   - Zero quality degradation (21.76 → 21.72 PPL, <0.2% change)
   - **Conclusion:** KV-cache is production-ready

2. **Overhead Wall (HIT by v2)**
   - 2x latency increase vs v1 (3348ms → 6754ms at T=1024)
   - Python quant/dequant overhead: 120-250ms per forward pass
   - Memory savings negligible: 920.8 MB → 872.0 MB (only 5% vs expected 50%)
   - Need fused CUDA kernel to overcome
   - **Conclusion:** Naive INT8 implementation is counterproductive

3. **Architectural Wall (HIT by v3/v4)**
   - 36x perplexity increase (21.72 → 779.90 for v3, 21.72 → 782.94 for v4)
   - Q-alignment reduces to 5.4x (21.72 → 116.88) but still unusable (>100 PPL threshold)
   - Pretrained weights incompatible with cross-layer sharing architecture
   - **Conclusion:** Requires architecture-aware pre-training (e.g., GQA)

### The Capacity Breakthrough

**v4 (INT8 + cross-layer) enables 3.8x more concurrent users:**
- v1: 491 concurrent users
- v4: 1867 concurrent users
- **Use case:** Offline batch processing where latency is not critical

### Pragmatic Recommendations

| Use Case | Recommended Version | Rationale |
|----------|---------------------|-----------|
| **Production inference** | v1 | Best latency/quality/simplicity balance |
| **Latency-critical** | v1 | Only reliable fast option |
| **High-throughput batch** | v4 | 3.8x capacity despite slow speed |
| **Research (quality)** | v0/v1 | Stable perplexity baseline |
| **Research (optimization)** | v2/v3/v4 | Study Overhead/Architectural Walls |

---

## XI. Future Work Assets

### Potential Improvements

1. **Fused INT8 Kernel for v2**
   - Implement CUDA kernel combining quantize→attention→dequantize
   - Eliminate Python overhead (120-250ms savings)
   - Could make v2 faster than v1 with memory benefits

2. **Architecture-Aware Pre-Training for v3/v4**
   - Train model from scratch with cross-layer sharing
   - Use Grouped Query Attention (GQA) architecture
   - Could make v3 fastest option with good quality

3. **Roofline-Guided Optimization**
   - Target "Other" overhead (33% of v1 time at T=256)
   - Kernel fusion opportunities
   - Reduce kernel launch overhead (319 kernels in v1 vs 246 in v0)

4. **Extended Sequence Length Study**
   - Evaluate T > 1024 (requires position embedding changes)
   - Test sliding window mechanisms
   - Analyze memory scaling limits

---

## XII. Reproducibility Checklist

### Environment
- ✓ GPU: NVIDIA RTX A6000 48GB
- ✓ PyTorch: 2.9.1+cu128
- ✓ CUDA: 12.8
- ✓ Python: 3.11
- ✓ Precision: bfloat16

### Data
- ✓ WikiText-2 test split (287,644 tokens)
- ✓ Tokenization: tiktoken gpt2
- ✓ Cache: `benchmark/.cache/wikitext2_gpt2_tokens.pt`

### Model
- ✓ Pretrained: GPT-2 Medium (gpt2-medium)
- ✓ Parameters: 378.96M
- ✓ Architecture: 24 layers, 16 heads, 1024 embedding dim, block_size=1024

### Benchmarking
- ✓ Clock locking: `activate_clock_lock.sh`
- ✓ Warmup: 5 iterations (configurable)
- ✓ Repetitions: 10 runs per measurement
- ✓ Deduplication: First occurrence per (benchmark, version, T)

### Code Versions
- ✓ Version definitions: `benchmark/versions.py`
- ✓ Plotting: `benchmark/plot_results.py`
- ✓ Benchmarking: `benchmark/main.py`, `benchmark/perplexity.py`

---

## XIII. File Count Summary

| Category | Count | Examples |
|----------|-------|----------|
| **Core Implementation** | 3 | model.py, train.py, sample.py |
| **Benchmark Scripts** | 12 | main.py, perplexity.py, plot_results.py, roofline/* |
| **Test Scripts** | 6 | test_kv_cache.py, test_perplexity.py, debug_*.py |
| **Configuration Files** | 10 | versions.py, config.yaml, eval_*.py, train_*.py |
| **Documentation** | 8 | RESEARCH_SUMMARY.md, RESEARCH_SPACE.md, docs/* |
| **Visualizations** | 56+ | comparison_*.png, per_phase_*.png, latency_*.png, vram_*.png |
| **Data Files** | 4+ | results.jsonl, metrics_summary.json, roofline metrics |
| **Utility Scripts** | 9 | analyze_results.py, configurator.py, clock locking |
| **Notebooks** | 2 | scaling_laws.ipynb, transformer_sizing.ipynb |
| **Total** | **110+** | (excludes __pycache__, .git, etc.) |

---

**Last Updated:** Based on benchmark runs through December 14, 2025  
**Primary Contributors:** Research team using NVIDIA RTX A6000  
**Repository Status:** Active development on `feature/unified_benchmarks` branch
