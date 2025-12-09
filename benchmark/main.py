# benchmark/main.py
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

# --- Patched Model for Per-phase Timing ---
def _forward_stepwise_for_timing(model, idx, timings):
    """Forward pass mirroring GPT.forward, with CUDA event timings per phase."""
    device = idx.device
    b, t = idx.size()
    assert t <= model.config.block_size, "Sequence too long"
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    # Embedding
    emb_start = torch.cuda.Event(enable_timing=True)
    emb_end = torch.cuda.Event(enable_timing=True)
    emb_start.record()
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    emb_end.record()
    timings['embedding'].append((emb_start, emb_end))

    # Blocks
    for block in model.transformer.h:
        # Attention
        attn_start = torch.cuda.Event(enable_timing=True)
        attn_end = torch.cuda.Event(enable_timing=True)
        attn_start.record()
        x = x + block.attn(block.ln_1(x))
        attn_end.record()
        timings['attention'].append((attn_start, attn_end))

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

    return logits, None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_result(config, result):
    """
    Appends a result record to the JSONL log file.
    """
    record = {
        'implementation_name': 'baseline',
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_total_vram': torch.cuda.get_device_properties(0).total_memory,
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'dtype': config['dtype'],
        'model_config': config['model_config'],
        'prompt_length': config['prompt_length'],
        'batch_size': config['batch_size'],
        'seed': config['seed'],
        'pretrained': config.get('pretrained', None),
        **result
    }
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    with open(config['log_file'], 'a') as f:
        f.write(json.dumps(record) + '\n')

def load_model(model_config, dtype_str, pretrained=None):
    """
    Loads the nanoGPT model on the GPU.
    """
    if pretrained:
        model = GPT.from_pretrained(pretrained)
    else:
        config = GPTConfig(**model_config)
        model = GPT(config)
    model.to("cuda")

    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    model.to(pt_dtype)

    model.eval()
    return model

def create_prompt(config):
    """
    Creates a synthetic prompt.
    """
    # Note: seed is set globally, but also locally here for clarity
    torch.manual_seed(config['seed'])
    prompt = torch.randint(
        0, config['model_config']['vocab_size'],
        size=(config['batch_size'], config['prompt_length']),
        device="cuda"
    )
    return prompt

@torch.no_grad()
def decode_loop(model, prompt, T):
    """
    The shared decode-loop runner.
    """
    idx = prompt
    for _ in range(T):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _, _ = model(idx_cond)
        # pluck the logits at the final step
        logits = logits[:, -1, :]
        # apply softmax to convert logits to (normalized) probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def run_latency_vs_T(config):
    print("Running Latency vs T benchmark...")
    model = load_model(config['model_config'], config['dtype'], config.get('pretrained'))
    prompt = create_prompt(config)

    for T in config['T_values']:
        print(f"  Benchmarking T={T}")
        # Warmup runs
        for _ in range(config['num_warmup_runs']):
            decode_loop(model, prompt, T)

        # Measured runs
        run_times = []
        for _ in range(config['num_measure_runs']):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            decode_loop(model, prompt, T)
            end_event.record()
            
            torch.cuda.synchronize()
            run_times.append(start_event.elapsed_time(end_event))

        time_total_ms_median = torch.median(torch.tensor(run_times)).item()
        time_total_ms_mean = torch.mean(torch.tensor(run_times)).item()
        time_total_ms_std = torch.std(torch.tensor(run_times)).item()
        time_per_token_ms_median = time_total_ms_median / T
        
        result = {
            'benchmark_name': 'latency_vs_T',
            'T': T,
            'time_total_ms_median': time_total_ms_median,
            'time_total_ms_mean': time_total_ms_mean,
            'time_total_ms_std': time_total_ms_std,
            'time_per_token_ms_median': time_per_token_ms_median,
        }
        log_result(config, result)
        print(f"    Median total time: {time_total_ms_median:.2f} ms")
        print(f"    Median time per token: {time_per_token_ms_median:.4f} ms")

def run_vram_vs_T(config):
    print("Running VRAM vs T benchmark...")
    model = load_model(config['model_config'], config['dtype'], config.get('pretrained'))
    prompt = create_prompt(config)

    for T in config['T_values']:
        print(f"  Benchmarking T={T}")
        
        # Reset memory stats and sync
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Baseline allocation with model on device (weights, buffers)
        baseline_alloc_bytes = torch.cuda.memory_allocated()

        # Measured run
        decode_loop(model, prompt, T)
        torch.cuda.synchronize()

        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_activation_bytes = max(0, peak_memory_bytes - baseline_alloc_bytes)
        
        result = {
            'benchmark_name': 'vram_vs_T',
            'T': T,
            'peak_memory_bytes': peak_memory_bytes,
            'peak_activation_bytes': peak_activation_bytes,
        }
        log_result(config, result)
        print(f"    Peak memory: {peak_memory_bytes / 1e9:.2f} GB")

def run_per_phase_timing(config):
    print("Running Per-phase timing benchmark...")

    # Load the same model as other benchmarks (incl. pretrained if set)
    model = load_model(config['model_config'], config['dtype'], config.get('pretrained'))
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    model.to(pt_dtype)
    model.eval()

    # --- Stepwise decode using the same model for timing ---
    @torch.no_grad()
    def decode_loop_stepwise(model, prompt, T, timings):
        idx = prompt
        for _ in range(T):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = _forward_stepwise_for_timing(model, idx_cond, timings)
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    prompt = create_prompt(config)
    T = config['T_star']

    # Warmup
    for _ in range(config['num_warmup_runs']):
        timings = {'embedding': [], 'attention': [], 'mlp': [], 'head': []}
        decode_loop_stepwise(model, prompt, T, timings)

    # Measurement
    all_run_phase_times = []
    for _ in range(config['num_measure_runs']):
        timings = {'embedding': [], 'attention': [], 'mlp': [], 'head': []}
        
        torch.cuda.synchronize()
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)

        total_start.record()
        decode_loop_stepwise(model, prompt, T, timings)
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
        'T_star': T,
        'time_phase_ms': median_phase_times
    }
    log_result(config, result)
    
    print(f"  T* = {T}")
    for phase, t in median_phase_times.items():
        print(f"    Median {phase} time: {t:.2f} ms")

def main():
    parser = argparse.ArgumentParser(description="nanoGPT benchmarking script.")
    parser.add_argument('benchmark', nargs='*', help="Benchmark(s) to run", default=['all'])
    parser.add_argument('--config', type=str, default='benchmark/config.yaml', help='Path to the config file.')
    parser.add_argument('--clear-log', action='store_true', help='Clear the results log file before running benchmarks.')
    parser.add_argument('--preset', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Use a pretrained GPT-2 preset for the model.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply preset from CLI or config to use pretrained weights if requested
    if args.preset:
        config['pretrained'] = args.preset
    else:
        # honor config file if it already specified pretrained
        config['pretrained'] = config.get('pretrained', 'gpt2')  # default to gpt2 per user request

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

if __name__ == '__main__':
    main()
