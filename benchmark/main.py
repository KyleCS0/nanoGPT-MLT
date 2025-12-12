# benchmark/main.py
"""
Main benchmarking script for nanoGPT KV-cache optimization variants.

Supports all versions defined in benchmark/versions.py:
- v0: No cache (baseline)
- v1: KV-cache enabled
- v2: KV-cache + INT8 quantization
- v3: KV-cache + cross-layer sharing
- v4: KV-cache + INT8 + cross-layer sharing

Usage:
    python benchmark/main.py                    # Run all benchmarks with v0, v1
    python benchmark/main.py --version v0 v1 v2 v3 v4  # All versions
    python benchmark/main.py latency --version v1 v2   # Compare v1 vs v2 latency
    python benchmark/main.py --preset gpt2-medium      # Use larger model
"""
import argparse
import yaml
import torch
import os
import platform
import json
import time
import random
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig, Block
from benchmark.versions import (
    VERSIONS, get_version_config, load_model_for_version,
    get_use_cache, get_cross_layer_sharing, get_model_config_dict
)

# --- Patched Model for Per-phase Timing ---
def _forward_stepwise_for_timing(model, idx, timings, past_key_values=None, use_cache=False, cross_layer_sharing=False):
    """
    Forward pass mirroring GPT.forward, with CUDA event timings per phase.

    Supports KV-cache and cross-layer sharing for accurate timing of all versions.
    """
    device = idx.device
    b, t = idx.size()

    # Determine if we are using cross-layer sharing
    use_cls = cross_layer_sharing if cross_layer_sharing is not None else model.config.cross_layer_sharing

    # Calculate position offset from cache
    past_length = 0
    if past_key_values is not None:
        if model.config.kv_cache_quant:
            past_length = past_key_values[0][0][0].size(2)
        else:
            past_length = past_key_values[0][0].size(2)

    total_length = past_length + t
    assert total_length <= model.config.block_size, "Sequence too long"
    pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)

    # Embedding
    emb_start = torch.cuda.Event(enable_timing=True)
    emb_end = torch.cuda.Event(enable_timing=True)
    emb_start.record()
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    emb_end.record()
    timings['embedding'].append((emb_start, emb_end))

    # Process through transformer blocks with KV-cache support
    present_key_values = [] if use_cache else None
    for i, block in enumerate(model.transformer.h):
        if use_cls:
            cache_idx = i // 2
            is_cache_owner = (i % 2 == 0)
            layer_past = past_key_values[cache_idx] if past_key_values is not None else None
        else:
            is_cache_owner = True
            layer_past = past_key_values[i] if past_key_values is not None else None

        # Attention
        attn_start = torch.cuda.Event(enable_timing=True)
        attn_end = torch.cuda.Event(enable_timing=True)
        attn_start.record()
        attn_out, present = block.attn(block.ln_1(x), layer_past=layer_past, use_cache=use_cache, is_cache_owner=is_cache_owner)
        x = x + attn_out
        attn_end.record()
        timings['attention'].append((attn_start, attn_end))

        if use_cache:
            if not use_cls or is_cache_owner:
                present_key_values.append(present)

        # MLP
        mlp_start = torch.cuda.Event(enable_timing=True)
        mlp_end = torch.cuda.Event(enable_timing=True)
        mlp_start.record()
        x = x + block.mlp(block.ln_2(x))
        mlp_end.record()
        timings['mlp'].append((mlp_start, mlp_end))

    x = model.transformer.ln_f(x)

    # Head (project only last time step)
    head_start = torch.cuda.Event(enable_timing=True)
    head_end = torch.cuda.Event(enable_timing=True)
    head_start.record()
    logits = model.lm_head(x[:, [-1], :])
    head_end.record()
    timings['head'].append((head_start, head_end))

    return logits, present_key_values

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_result(config, result, model_config=None):
    """
    Appends a result record to the JSONL log file.

    Args:
        config: Benchmark configuration dict
        result: Benchmark result dict
        model_config: Optional model config dict (for pretrained models)
    """
    # Use provided model_config or fall back to config's model_config
    actual_model_config = model_config if model_config else config.get('model_config', {})

    record = {
        'implementation_name': 'baseline',
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_total_vram': torch.cuda.get_device_properties(0).total_memory,
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'dtype': config['dtype'],
        'model_config': actual_model_config,
        'prompt_length': config['prompt_length'],
        'batch_size': config['batch_size'],
        'seed': config['seed'],
        'pretrained': config.get('pretrained', None),
        **result
    }
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    with open(config['log_file'], 'a') as f:
        f.write(json.dumps(record) + '\n')


def create_prompt(config, vocab_size=50257):
    """
    Creates a synthetic prompt.

    Args:
        config: Benchmark configuration dict
        vocab_size: Vocabulary size (default: GPT-2's 50257)
    """
    # Note: seed is set globally, but also locally here for clarity
    torch.manual_seed(config['seed'])
    prompt = torch.randint(
        0, vocab_size,
        size=(config['batch_size'], config['prompt_length']),
        device="cuda"
    )
    return prompt


@torch.no_grad()
def decode_loop(model, prompt, T, use_cache=False, cross_layer_sharing=None):
    """
    The shared decode-loop runner.

    Args:
        model: The GPT model
        prompt: Input prompt tensor (B, prompt_len)
        T: Number of tokens to generate
        use_cache: If True, use KV-cache for efficient generation
        cross_layer_sharing: If True/False, override model's cross_layer_sharing setting
                             If None, use model's config setting
    """
    # Use the model's generate method which handles caching properly
    return model.generate(prompt, max_new_tokens=T, use_cache=use_cache,
                          cross_layer_sharing=cross_layer_sharing)

def remove_outliers_iqr(data, k=1.5):
    """
    Remove outliers using IQR method.

    Args:
        data: List or tensor of values
        k: IQR multiplier (1.5 = standard, 3.0 = extreme only)

    Returns:
        Filtered tensor with outliers removed
    """
    tensor = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    mask = (tensor >= lower_bound) & (tensor <= upper_bound)
    return tensor[mask]


def run_latency_vs_T(config):
    """
    Benchmark latency vs generation length T for each version.

    For each version, loads a model with the appropriate settings and measures
    generation time across different T values.
    """
    print("Running Latency vs T benchmark...")

    # Get versions to benchmark (default to v0 and v1)
    versions = config.get('versions', ['v0', 'v1'])
    pretrained = config.get('pretrained', 'gpt2')
    dtype_str = config['dtype']

    for version in versions:
        # Load model configured for this version
        model, v_config = load_model_for_version(version, pretrained, dtype_str)
        model_config_dict = get_model_config_dict(model)
        prompt = create_prompt(config, vocab_size=model.config.vocab_size)

        use_cache = v_config['use_cache']
        cross_layer_sharing = v_config['cross_layer_sharing']
        desc = v_config['description']

        print(f"\n  Version: {version} ({desc})")
        print(f"    use_cache={use_cache}, kv_cache_quant={v_config['kv_cache_quant']}, cross_layer_sharing={cross_layer_sharing}")

        for T in config['T_values']:
            print(f"    Benchmarking T={T}")

            # Clear CUDA cache and synchronize before each T value
            # This prevents memory fragmentation from affecting measurements
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Warmup runs - more warmup for larger T values
            num_warmup = config['num_warmup_runs']
            if T > 512:
                num_warmup = max(num_warmup, 15)  # Extra warmup for large T

            for _ in range(num_warmup):
                decode_loop(model, prompt.clone(), T, use_cache=use_cache,
                           cross_layer_sharing=cross_layer_sharing)
                torch.cuda.synchronize()

            # Clear cache after warmup to start measurements fresh
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Measured runs
            run_times = []
            for _ in range(config['num_measure_runs']):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                decode_loop(model, prompt.clone(), T, use_cache=use_cache,
                           cross_layer_sharing=cross_layer_sharing)
                end_event.record()

                torch.cuda.synchronize()
                run_times.append(start_event.elapsed_time(end_event))

            # Remove outliers using IQR method for more robust statistics
            run_times_tensor = torch.tensor(run_times)
            run_times_filtered = remove_outliers_iqr(run_times_tensor, k=1.5)

            # Fall back to original if too many removed (>50%)
            if len(run_times_filtered) < len(run_times) // 2:
                print(f"      Warning: IQR removed >50% of samples, using all data")
                run_times_filtered = run_times_tensor

            n_outliers = len(run_times) - len(run_times_filtered)

            time_total_ms_median = torch.median(run_times_filtered).item()
            time_total_ms_mean = torch.mean(run_times_filtered).item()
            time_total_ms_std = torch.std(run_times_filtered).item() if len(run_times_filtered) > 1 else 0.0
            time_per_token_ms_median = time_total_ms_median / T
            time_per_token_ms_std = time_total_ms_std / T

            result = {
                'benchmark_name': 'latency_vs_T',
                'version': version,
                'version_description': desc,
                'use_cache': use_cache,
                'kv_cache_quant': v_config['kv_cache_quant'],
                'cross_layer_sharing': cross_layer_sharing,
                'T': T,
                'time_total_ms_median': time_total_ms_median,
                'time_total_ms_mean': time_total_ms_mean,
                'time_total_ms_std': time_total_ms_std,
                'time_per_token_ms_median': time_per_token_ms_median,
                'time_per_token_ms_std': time_per_token_ms_std,
                'n_samples': len(run_times_filtered),
                'n_outliers_removed': n_outliers,
            }
            log_result(config, result, model_config_dict)
            print(f"      Median total time: {time_total_ms_median:.2f} ms (std={time_total_ms_std:.2f}, outliers={n_outliers})")
            print(f"      Median time per token: {time_per_token_ms_median:.4f} ms")

        # Clean up model to free memory before loading next version
        del model
        torch.cuda.empty_cache()

def run_vram_vs_T(config):
    """
    Benchmark VRAM usage vs generation length T for each version.

    Measures peak memory and activation memory for each version configuration.
    """
    print("Running VRAM vs T benchmark...")

    # Get versions to benchmark (default to v0 and v1)
    versions = config.get('versions', ['v0', 'v1'])
    pretrained = config.get('pretrained', 'gpt2')
    dtype_str = config['dtype']

    for version in versions:
        # Load model configured for this version
        model, v_config = load_model_for_version(version, pretrained, dtype_str)
        model_config_dict = get_model_config_dict(model)
        prompt = create_prompt(config, vocab_size=model.config.vocab_size)

        use_cache = v_config['use_cache']
        cross_layer_sharing = v_config['cross_layer_sharing']
        desc = v_config['description']

        print(f"\n  Version: {version} ({desc})")
        print(f"    use_cache={use_cache}, kv_cache_quant={v_config['kv_cache_quant']}, cross_layer_sharing={cross_layer_sharing}")

        for T in config['T_values']:
            print(f"    Benchmarking T={T}")

            # Reset memory stats and sync
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Baseline allocation with model on device (weights, buffers)
            baseline_alloc_bytes = torch.cuda.memory_allocated()

            # Measured run
            decode_loop(model, prompt.clone(), T, use_cache=use_cache,
                       cross_layer_sharing=cross_layer_sharing)
            torch.cuda.synchronize()

            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_activation_bytes = max(0, peak_memory_bytes - baseline_alloc_bytes)

            result = {
                'benchmark_name': 'vram_vs_T',
                'version': version,
                'version_description': desc,
                'use_cache': use_cache,
                'kv_cache_quant': v_config['kv_cache_quant'],
                'cross_layer_sharing': cross_layer_sharing,
                'T': T,
                'peak_memory_bytes': peak_memory_bytes,
                'peak_activation_bytes': peak_activation_bytes,
                'baseline_memory_bytes': baseline_alloc_bytes,
            }
            log_result(config, result, model_config_dict)
            print(f"      Peak memory: {peak_memory_bytes / 1e9:.2f} GB")
            print(f"      Activation memory: {peak_activation_bytes / 1e6:.2f} MB")

        # Clean up model to free memory before loading next version
        del model
        torch.cuda.empty_cache()

def run_per_phase_timing(config):
    """
    Benchmark per-phase timing (embedding, attention, MLP, head) for each version.

    Uses a custom forward pass that times each phase separately while supporting
    KV-cache and cross-layer sharing for accurate timing of all versions.
    """
    print("Running Per-phase timing benchmark...")

    # Get versions to benchmark (default to v0 and v1)
    versions = config.get('versions', ['v0', 'v1'])
    pretrained = config.get('pretrained', 'gpt2')
    dtype_str = config['dtype']
    T = config['T_star']

    for version in versions:
        # Load model configured for this version
        model, v_config = load_model_for_version(version, pretrained, dtype_str)
        model_config_dict = get_model_config_dict(model)
        prompt = create_prompt(config, vocab_size=model.config.vocab_size)

        use_cache = v_config['use_cache']
        cross_layer_sharing = v_config['cross_layer_sharing']
        desc = v_config['description']

        print(f"\n  Version: {version} ({desc})")
        print(f"    use_cache={use_cache}, kv_cache_quant={v_config['kv_cache_quant']}, cross_layer_sharing={cross_layer_sharing}")

        # --- Stepwise decode with timing, supporting KV-cache ---
        @torch.no_grad()
        def decode_loop_stepwise(model, prompt, T, timings, use_cache, cross_layer_sharing):
            idx = prompt
            past_key_values = None

            for step in range(T):
                if use_cache and past_key_values is not None:
                    # Cached: only feed the last token
                    idx_cond = idx[:, -1:]
                else:
                    # Non-cached or first step: feed full sequence (up to block_size)
                    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

                logits, past_key_values = _forward_stepwise_for_timing(
                    model, idx_cond, timings,
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache,
                    cross_layer_sharing=cross_layer_sharing
                )

                probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

                # Handle cache overflow
                if use_cache and past_key_values is not None:
                    if model.config.kv_cache_quant:
                        cache_len = past_key_values[0][0][0].size(2)
                    else:
                        cache_len = past_key_values[0][0].size(2)
                    if cache_len >= model.config.block_size:
                        # Trim cache
                        if model.config.kv_cache_quant:
                            past_key_values = [
                                ((k[0][:, :, -(model.config.block_size - 1):, :], k[1]),
                                 (v[0][:, :, -(model.config.block_size - 1):, :], v[1]))
                                for k, v in past_key_values
                            ]
                        else:
                            past_key_values = [
                                (k[:, :, -(model.config.block_size - 1):, :],
                                 v[:, :, -(model.config.block_size - 1):, :])
                                for k, v in past_key_values
                            ]

            return idx

        # Warmup
        for _ in range(config['num_warmup_runs']):
            timings = {'embedding': [], 'attention': [], 'mlp': [], 'head': []}
            decode_loop_stepwise(model, prompt.clone(), T, timings, use_cache, cross_layer_sharing)

        # Measurement
        all_run_phase_times = []
        for _ in range(config['num_measure_runs']):
            timings = {'embedding': [], 'attention': [], 'mlp': [], 'head': []}

            torch.cuda.synchronize()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)

            total_start.record()
            decode_loop_stepwise(model, prompt.clone(), T, timings, use_cache, cross_layer_sharing)
            total_end.record()
            torch.cuda.synchronize()

            phase_times = {phase: 0.0 for phase in timings}
            for phase, events in timings.items():
                for start, end in events:
                    phase_times[phase] += start.elapsed_time(end)

            phase_times['total'] = total_start.elapsed_time(total_end)
            all_run_phase_times.append(phase_times)

        # Aggregate results
        median_phase_times = {}
        for phase in all_run_phase_times[0].keys():
            median_phase_times[phase] = torch.median(torch.tensor([run[phase] for run in all_run_phase_times])).item()

        # Calculate "other"
        measured_sum = median_phase_times['embedding'] + median_phase_times['attention'] + median_phase_times['mlp'] + median_phase_times['head']
        median_phase_times['other'] = median_phase_times['total'] - measured_sum

        result = {
            'benchmark_name': 'per_phase_timing',
            'version': version,
            'version_description': desc,
            'use_cache': use_cache,
            'kv_cache_quant': v_config['kv_cache_quant'],
            'cross_layer_sharing': cross_layer_sharing,
            'T_star': T,
            'time_phase_ms': median_phase_times
        }
        log_result(config, result, model_config_dict)

        print(f"    T* = {T}")
        for phase, t in median_phase_times.items():
            print(f"      Median {phase} time: {t:.2f} ms")

        # Clean up model to free memory before loading next version
        del model
        torch.cuda.empty_cache()

def main():
    # Build version choices and help from registry
    available_versions = list(VERSIONS.keys())
    version_help = ', '.join(f"{v} ({VERSIONS[v]['description']})" for v in available_versions)

    parser = argparse.ArgumentParser(
        description="nanoGPT benchmarking script for KV-cache optimization variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark/main.py                           # Run all benchmarks with v0, v1
    python benchmark/main.py --version v0 v1 v2 v3 v4  # All versions
    python benchmark/main.py latency --version v1 v2   # Compare v1 vs v2 latency
    python benchmark/main.py --preset gpt2-medium      # Use larger model
    python benchmark/main.py vram --version v1 v3 v4   # Compare memory optimization versions
        """
    )
    parser.add_argument('benchmark', nargs='*', help="Benchmark(s) to run: latency, vram, phase, all", default=['all'])
    parser.add_argument('--config', type=str, default='benchmark/config.yaml', help='Path to the config file.')
    parser.add_argument('--clear-log', action='store_true', help='Clear the results log file before running benchmarks.')
    parser.add_argument('--preset', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Use a pretrained GPT-2 preset for the model.')
    parser.add_argument('--version', type=str, nargs='+', default=None, choices=available_versions,
                        help=f'Version(s) to benchmark: {version_help}. Default: v0, v1.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply preset from CLI or config to use pretrained weights if requested
    if args.preset:
        config['pretrained'] = args.preset
    else:
        # honor config file if it already specified pretrained
        config['pretrained'] = config.get('pretrained', 'gpt2')  # default to gpt2 per user request

    # Apply version(s) from CLI (validated against registry)
    if args.version:
        config['versions'] = args.version
    elif 'versions' not in config:
        config['versions'] = ['v0', 'v1']  # default to baseline comparisons

    # Validate versions
    for v in config['versions']:
        if v not in VERSIONS:
            print(f"Error: Unknown version '{v}'. Available: {available_versions}")
            sys.exit(1)

    # Print configuration summary
    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print(f"  Model: {config['pretrained']}")
    print(f"  Versions: {', '.join(config['versions'])}")
    print(f"  Benchmarks: {', '.join(args.benchmark)}")
    print(f"  dtype: {config['dtype']}")
    print("=" * 60)

    # Optionally clear the results log
    if args.clear_log:
        log_path = config['log_file']
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("")
        print(f"Cleared log: {log_path}")

    set_seed(config['seed'])

    if 'all' in args.benchmark or 'latency' in args.benchmark:
        run_latency_vs_T(config)
    if 'all' in args.benchmark or 'vram' in args.benchmark:
        run_vram_vs_T(config)
    if 'all' in args.benchmark or 'phase' in args.benchmark:
        run_per_phase_timing(config)

    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print(f"Results saved to: {config['log_file']}")
    print("=" * 60)

if __name__ == '__main__':
    main()
