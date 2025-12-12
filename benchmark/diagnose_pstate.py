#!/usr/bin/env python3
"""
Diagnostic script to verify GPU P-state switching hypothesis.

This script:
1. Monitors GPU clock speeds during idle and load
2. Simulates the benchmark pattern (work -> idle -> work)
3. Shows if clock speeds vary between measurements
"""

import torch
import time
import subprocess
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig


def get_gpu_clocks():
    """Query current GPU clocks via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'graphics_mhz': int(parts[0]),
                'memory_mhz': int(parts[1]),
                'power_w': float(parts[2]),
                'temp_c': int(parts[3])
            }
    except Exception as e:
        print(f"Warning: Could not query nvidia-smi: {e}")
    return None


def run_workload(model, prompt, T, use_cache=False):
    """Run a single generation workload."""
    with torch.no_grad():
        return model.generate(prompt, max_new_tokens=T, use_cache=use_cache)


def measure_with_clocks(model, prompt, T, use_cache, label):
    """Measure latency and record GPU clocks before/during/after."""
    torch.cuda.synchronize()

    clocks_before = get_gpu_clocks()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    run_workload(model, prompt, T, use_cache)
    end.record()

    torch.cuda.synchronize()

    clocks_after = get_gpu_clocks()
    elapsed_ms = start.elapsed_time(end)

    return {
        'label': label,
        'T': T,
        'elapsed_ms': elapsed_ms,
        'ms_per_token': elapsed_ms / T,
        'clocks_before': clocks_before,
        'clocks_after': clocks_after
    }


def main():
    print("=" * 70)
    print("GPU P-State Diagnostic")
    print("=" * 70)

    # Check initial GPU state
    initial_clocks = get_gpu_clocks()
    if initial_clocks:
        print(f"\nInitial GPU state (idle):")
        print(f"  Graphics: {initial_clocks['graphics_mhz']} MHz")
        print(f"  Memory:   {initial_clocks['memory_mhz']} MHz")
        print(f"  Power:    {initial_clocks['power_w']} W")
        print(f"  Temp:     {initial_clocks['temp_c']} C")

    # Create model
    print("\nLoading model...")
    config = GPTConfig(
        n_layer=12, n_head=12, n_embd=768,
        block_size=1024, vocab_size=50257
    )
    model = GPT(config).cuda().bfloat16()
    model.eval()

    prompt = torch.randint(0, 50257, (1, 32), device='cuda')

    # Test 1: Measure clock behavior with varying idle gaps
    print("\n" + "=" * 70)
    print("Test 1: Effect of idle time between measurements")
    print("=" * 70)

    T = 128
    idle_gaps = [0, 0.5, 1.0, 2.0, 5.0]  # seconds

    # Initial warmup
    print("\nWarming up GPU (10 iterations)...")
    for _ in range(10):
        run_workload(model, prompt, T, use_cache=False)
        torch.cuda.synchronize()

    for gap in idle_gaps:
        print(f"\n--- Idle gap: {gap}s ---")

        # Wait for idle period
        if gap > 0:
            time.sleep(gap)

        # Measure
        result = measure_with_clocks(model, prompt, T, use_cache=False, label=f"gap_{gap}s")

        print(f"  Latency: {result['elapsed_ms']:.1f} ms ({result['ms_per_token']:.2f} ms/token)")
        if result['clocks_before']:
            print(f"  Clocks before: {result['clocks_before']['graphics_mhz']} MHz, {result['clocks_before']['power_w']}W")
        if result['clocks_after']:
            print(f"  Clocks after:  {result['clocks_after']['graphics_mhz']} MHz, {result['clocks_after']['power_w']}W")

    # Test 2: Simulate actual benchmark pattern
    print("\n" + "=" * 70)
    print("Test 2: Simulating benchmark measurement pattern")
    print("=" * 70)

    T_values = [128, 160, 256, 320, 512]
    results = []

    for T in T_values:
        print(f"\n--- T={T} ---")

        # Simulate benchmark: clear cache, warmup, measure
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Brief warmup (as in your benchmark)
        for _ in range(3):
            run_workload(model, prompt, T, use_cache=False)
            torch.cuda.synchronize()

        # Measure multiple times
        times = []
        clocks = []
        for i in range(5):
            result = measure_with_clocks(model, prompt, T, use_cache=False, label=f"T{T}_run{i}")
            times.append(result['ms_per_token'])
            if result['clocks_after']:
                clocks.append(result['clocks_after']['graphics_mhz'])

        print(f"  Per-token latencies: {[f'{t:.2f}' for t in times]} ms")
        print(f"  Clock speeds:        {clocks} MHz")
        print(f"  Latency range: {min(times):.2f} - {max(times):.2f} ms (spread: {max(times)-min(times):.2f})")

        results.append({
            'T': T,
            'times': times,
            'clocks': clocks
        })

    # Test 3: Continuous load vs interrupted load
    print("\n" + "=" * 70)
    print("Test 3: Continuous vs interrupted workload")
    print("=" * 70)

    T = 256

    # Continuous: back-to-back runs
    print("\nContinuous (no gaps):")
    continuous_times = []
    continuous_clocks = []
    for i in range(10):
        result = measure_with_clocks(model, prompt, T, use_cache=False, label=f"continuous_{i}")
        continuous_times.append(result['ms_per_token'])
        if result['clocks_after']:
            continuous_clocks.append(result['clocks_after']['graphics_mhz'])
    print(f"  Latencies: {[f'{t:.2f}' for t in continuous_times]} ms")
    print(f"  Clocks:    {continuous_clocks} MHz")
    print(f"  Mean: {sum(continuous_times)/len(continuous_times):.2f} ms, Std: {torch.std(torch.tensor(continuous_times)).item():.2f} ms")

    # Interrupted: 1s gap between runs
    print("\nInterrupted (1s gaps):")
    interrupted_times = []
    interrupted_clocks = []
    for i in range(10):
        time.sleep(1.0)
        result = measure_with_clocks(model, prompt, T, use_cache=False, label=f"interrupted_{i}")
        interrupted_times.append(result['ms_per_token'])
        if result['clocks_after']:
            interrupted_clocks.append(result['clocks_after']['graphics_mhz'])
    print(f"  Latencies: {[f'{t:.2f}' for t in interrupted_times]} ms")
    print(f"  Clocks:    {interrupted_clocks} MHz")
    print(f"  Mean: {sum(interrupted_times)/len(interrupted_times):.2f} ms, Std: {torch.std(torch.tensor(interrupted_times)).item():.2f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    cont_std = torch.std(torch.tensor(continuous_times)).item()
    int_std = torch.std(torch.tensor(interrupted_times)).item()
    cont_mean = sum(continuous_times) / len(continuous_times)
    int_mean = sum(interrupted_times) / len(interrupted_times)

    print(f"\nContinuous workload: mean={cont_mean:.2f} ms/token, std={cont_std:.2f}")
    print(f"Interrupted workload: mean={int_mean:.2f} ms/token, std={int_std:.2f}")

    if int_std > cont_std * 2:
        print("\n⚠️  HIGH VARIANCE with idle gaps - P-state switching CONFIRMED")
        print("    Recommendation: Lock GPU clocks or add continuous background load")
    elif int_std > cont_std * 1.5:
        print("\n⚠️  MODERATE VARIANCE with idle gaps - P-state switching LIKELY")
        print("    Recommendation: Consider locking GPU clocks")
    else:
        print("\n✓  Variance similar - P-state switching NOT the primary issue")
        print("    Look for other causes (thermal throttling, memory, etc.)")

    # Check for clock variation
    all_clocks = continuous_clocks + interrupted_clocks
    if all_clocks:
        clock_range = max(all_clocks) - min(all_clocks)
        if clock_range > 100:
            print(f"\n⚠️  GPU clock varied by {clock_range} MHz - confirms P-state switching")
        else:
            print(f"\n✓  GPU clock stable (range: {clock_range} MHz)")


if __name__ == '__main__':
    main()
