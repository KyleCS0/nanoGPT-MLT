# Roofline Analysis Benchmark

This directory contains tools to perform a Roofline Analysis on the nanoGPT model, specifically comparing the **Standard Attention (V0)** mechanism against the **KV-Cache Optimized (V1)** mechanism.

## Overview
The goal is to physically demonstrate that:
*   **V0 (No Cache)** is **Compute Bound** (or moves towards it) due to redundant $O(T^2)$ computations.
*   **V1 (KV Cache)** is **Memory Bound** because it minimizes compute to $O(T)$, leaving memory bandwidth as the bottleneck.

## Files
*   `profile_decode_step.py`: The PyTorch script that runs the specific decode step for profiling.
*   `run_ncu.sh`: Shell script to run NVIDIA Nsight Compute (`ncu`) with correct permissions and settings.
*   `parse_ncu_results.py`: Parsers the raw CSV output from `ncu` to extract FLOPs and DRAM bytes.
*   `plot_roofline.py`: Generates the standard Roofline visualization.
*   `analyze_results.sh`: End-to-end script that parses reports and runs the plotter.

## Usage

### 1. Run Profiling
**Requires `sudo`** to access GPU performance counters.
Adjust parameters (Model, Batch Size, Sequence Length) in `run_ncu.sh`.

```bash
sudo bash benchmark/roofline/run_ncu.sh
```
*Outputs: `ncu_reports/*.ncu-rep` (Massive binary files)*

### 2. Analyze & Plot
Parses the reports and generates `roofline.png`.
Adjust `BATCH_SIZE` and `T` in `analyze_results.sh` to match your run.

```bash
bash benchmark/roofline/analyze_results.sh
```
*Outputs: `benchmark/roofline/roofline.png`, `metrics/*.json`*

## Version Control Guidelines

### ✅ What to Commit
*   All Python scripts (`*.py`)
*   All Shell scripts (`*.sh`)
*   This `README.md`
*   (Optional) `roofline.png` if you want to snapshot the current state.

### ❌ What NOT to Commit
*   **`ncu_reports/*.ncu-rep`**: These files are **HUGE** (100MB - 1GB+). Do not add them to git.
*   `metrics/*.json`: Intermediate data files.
*   `__pycache__`
