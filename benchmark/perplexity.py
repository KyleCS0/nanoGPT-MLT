# benchmark/perplexity.py
"""
Perplexity evaluation for nanoGPT KV-cache optimization variants.

Measures model quality on WikiText-2 to verify optimizations don't degrade accuracy.

Two evaluation modes:
1. Teacher-forcing (default): Full sequence forward pass. Fast but doesn't use KV-cache,
   so all versions produce identical perplexity.
2. Autoregressive (--autoregressive): Processes one token at a time with KV-cache.
   Slower but actually tests cache optimizations (INT8 quantization, cross-layer sharing).

Usage:
    # Teacher-forcing mode (fast, but doesn't test cache)
    python benchmark/perplexity.py --version v0 v1

    # Autoregressive mode (slow, but properly tests cache optimizations)
    python benchmark/perplexity.py --autoregressive --version v0 v1 v2 v3 v4

    # Quick test with limited tokens
    python benchmark/perplexity.py --autoregressive --max-tokens 10000 --version v1 v2
"""

import argparse
import math
import os
import sys
import json
import platform
from tqdm import tqdm

import torch
import tiktoken

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig, KVCache
from benchmark.versions import VERSIONS, get_version_config, load_model_for_version, get_model_config_dict
import torch.nn.functional as F


def load_wikitext2(max_tokens=None):
    """
    Load WikiText-2 test set and tokenize with tiktoken.

    Args:
        max_tokens: Limit to first N tokens (for quick testing)

    Returns:
        tokens: 1D tensor of token IDs
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join(dataset["text"])

    # Tokenize with GPT-2 tokenizer
    print("Tokenizing with GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    print(f"Total tokens: {len(tokens):,}")
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def compute_perplexity(model, tokens, stride=512, device='cuda'):
    """
    Compute perplexity using sliding window evaluation.

    Uses the standard strided sliding window approach where overlapping tokens
    are used as context but only non-overlapping tokens contribute to the loss.
    This provides better perplexity estimates than naive chunking while being
    computationally tractable.

    Reference: https://huggingface.co/docs/transformers/perplexity

    Args:
        model: GPT model
        tokens: 1D tensor of token IDs
        stride: Sliding window stride (smaller = more context = better PPL)
        device: Device to run on

    Returns:
        perplexity: exp(avg_loss)
        avg_loss: Average cross-entropy loss per token
        total_tokens: Number of tokens evaluated (non-overlapping)
    """
    model.eval()
    block_size = model.config.block_size
    seq_len = len(tokens)

    nll_sum = 0.0  # Sum of negative log likelihoods
    total_tokens = 0
    prev_end = 0

    # Sliding window evaluation
    num_windows = max(1, (seq_len - 1 + stride - 1) // stride)
    pbar = tqdm(range(0, seq_len - 1, stride), desc="Evaluating", total=num_windows)

    for begin in pbar:
        end = min(begin + block_size, seq_len)

        # Input: tokens[begin:end-1], Target: tokens[begin+1:end]
        input_ids = tokens[begin:end-1].unsqueeze(0).to(device)
        target_ids = tokens[begin+1:end].to(device)

        # Forward pass - pass targets to get full sequence logits (not just last position)
        # The model optimizes inference by only returning last position when targets=None
        logits, _, _ = model(input_ids, targets=target_ids.unsqueeze(0))

        # Compute per-token cross entropy loss
        # logits: (1, seq, vocab), target_ids: (seq,)
        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

        # Only count loss for non-overlapping tokens (new tokens in this window)
        # trg_len = number of new tokens we haven't seen before
        trg_len = end - prev_end

        # Select only the rightmost trg_len tokens for loss computation
        # These are positions [-(trg_len):] in the sequence
        if trg_len > 0:
            # Get the target tokens for loss (last trg_len tokens)
            loss_targets = target_ids[-trg_len:]
            # Get corresponding log probs (last trg_len positions)
            loss_log_probs = log_probs[-trg_len:]

            # Gather the log prob of the correct token at each position
            token_log_probs = loss_log_probs.gather(1, loss_targets.unsqueeze(1)).squeeze(1)
            # Negative log likelihood
            nll_sum += -token_log_probs.sum().item()
            total_tokens += trg_len

        prev_end = end
        current_loss = nll_sum / total_tokens if total_tokens > 0 else 0
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        if end >= seq_len:
            break

    avg_loss = nll_sum / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss, total_tokens


@torch.no_grad()
def compute_perplexity_autoregressive(model, tokens, device='cuda', use_cache=True,
                                       cross_layer_sharing=False):
    """
    Compute perplexity using true autoregressive evaluation (one token at a time).

    This method processes tokens sequentially, using the KV-cache at each step.
    Unlike teacher forcing, this actually exercises the cache optimizations and
    will reveal any quality degradation from INT8 quantization or cross-layer sharing.

    Args:
        model: GPT model
        tokens: 1D tensor of token IDs
        device: Device to run on
        use_cache: Whether to use KV-cache (False for v0 baseline)
        cross_layer_sharing: Whether cross-layer sharing is enabled

    Returns:
        perplexity: exp(avg_loss)
        avg_loss: Average cross-entropy loss per token
        total_tokens: Number of tokens evaluated
    """
    model.eval()
    block_size = model.config.block_size

    # Initialize KV-cache (for v1-v4)
    kv_cache = None
    if use_cache:
        kv_cache = KVCache(
            batch_size=1,
            max_seq_len=block_size,
            n_heads=model.config.n_head,
            head_dim=model.config.n_embd // model.config.n_head,
            n_layers=model.config.n_layer,
            device=device,
            dtype=next(model.parameters()).dtype,
            cross_layer_sharing=cross_layer_sharing,
            kv_cache_quant=model.config.kv_cache_quant,
        )

    nll_sum = 0.0
    total_tokens = 0

    pbar = tqdm(range(len(tokens) - 1), desc="Evaluating (autoregressive)")

    for t in pbar:
        current_token = tokens[t:t+1].unsqueeze(0).to(device)  # [1, 1]
        target_token = tokens[t+1].to(device)  # scalar

        # Handle context overflow: trim cache if at block_size
        if kv_cache is not None and kv_cache.seq_len >= block_size:
            kv_cache.trim(block_size - 1)

        # Forward pass
        if use_cache and kv_cache is not None:
            logits, _, _ = model(
                current_token,
                use_cache=True,
                cross_layer_sharing=cross_layer_sharing,
                kv_cache=kv_cache
            )
        else:
            # v0: No cache - must pass full context each time (capped at block_size)
            ctx_start = max(0, t - block_size + 1)
            context = tokens[ctx_start:t+1].unsqueeze(0).to(device)
            logits, _, _ = model(context, use_cache=False)

        # Get logits for next token prediction (last position)
        next_logits = logits[0, -1, :]  # [vocab_size]

        # Cross-entropy loss
        loss = F.cross_entropy(next_logits.unsqueeze(0), target_token.unsqueeze(0))
        nll_sum += loss.item()
        total_tokens += 1

        # Update progress bar
        current_loss = nll_sum / total_tokens
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})

    avg_loss = nll_sum / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss, total_tokens


def log_result(result, log_file, model_config, preset, dtype):
    """
    Append result to JSONL log file.

    Args:
        result: Dict with benchmark results
        log_file: Path to log file
        model_config: Model configuration dict
        preset: Model preset name
        dtype: Data type string
    """
    record = {
        'implementation_name': 'baseline',
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_total_vram': torch.cuda.get_device_properties(0).total_memory,
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'dtype': dtype,
        'model_config': model_config,
        'pretrained': preset,
        **result
    }
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(json.dumps(record) + '\n')


def print_results_table(results, versions):
    """
    Print a formatted results table.

    Args:
        results: Dict mapping version -> perplexity
        versions: List of version names
    """
    # Find baseline (first version)
    baseline_version = versions[0]
    baseline_ppl = results[baseline_version]['perplexity']

    print("\n" + "=" * 70)
    print("Results:")
    print("-" * 70)
    print(f"{'Version':<10} {'Description':<30} {'Perplexity':>12} {'Degradation':>12}")
    print("-" * 70)

    for version in versions:
        ppl = results[version]['perplexity']
        v_config = get_version_config(version)
        desc = v_config['description']

        if version == baseline_version:
            deg_str = "baseline"
        else:
            degradation = ((ppl - baseline_ppl) / baseline_ppl) * 100
            deg_str = f"{degradation:+.2f}%"

        print(f"{version:<10} {desc:<30} {ppl:>12.2f} {deg_str:>12}")

    print("=" * 70)


def main():
    # Available versions from unified registry
    available_versions = list(VERSIONS.keys())
    version_help = ', '.join(f"{v} ({VERSIONS[v]['description']})" for v in available_versions)

    parser = argparse.ArgumentParser(
        description="Perplexity evaluation for nanoGPT KV-cache variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation modes:
  Default (teacher-forcing): Fast but doesn't use KV-cache. All versions identical.
  --autoregressive: Slow but tests cache. May show differences for INT8/cross-layer.

Example:
  python benchmark/perplexity.py --autoregressive --max-tokens 10000 --version v1 v2
        """
    )
    parser.add_argument(
        '--preset', type=str,
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
        default='gpt2',
        help='Pretrained GPT-2 model to use'
    )
    parser.add_argument(
        '--version', type=str, nargs='+',
        default=['v0', 'v1'],
        choices=available_versions,
        help=f'Version(s) to evaluate: {version_help}. Default: v0, v1.'
    )
    parser.add_argument(
        '--stride', type=int, default=512,
        help='Sliding window stride (default: 512)'
    )
    parser.add_argument(
        '--max-tokens', type=int, default=None,
        help='Limit evaluation to first N tokens (for quick testing)'
    )
    parser.add_argument(
        '--dtype', type=str, default='bfloat16',
        choices=['float32', 'bfloat16', 'float16'],
        help='Data type for model'
    )
    parser.add_argument(
        '--log-file', type=str, default='benchmark/results.jsonl',
        help='Path to results log file'
    )
    parser.add_argument(
        '--clear-log', action='store_true',
        help='Clear the results log before running'
    )
    parser.add_argument(
        '--autoregressive', action='store_true',
        help='Use autoregressive evaluation (one token at a time with KV-cache). '
             'Slower but actually tests cache optimizations.'
    )

    args = parser.parse_args()

    # Validate versions
    for v in args.version:
        if v not in VERSIONS:
            print(f"Error: Unknown version '{v}'. Available: {available_versions}")
            sys.exit(1)

    # Clear log if requested
    if args.clear_log and os.path.exists(args.log_file):
        os.remove(args.log_file)
        print(f"Cleared log: {args.log_file}")

    # Load dataset
    tokens = load_wikitext2(max_tokens=args.max_tokens)

    eval_mode = "autoregressive" if args.autoregressive else "teacher-forcing"

    print(f"\n" + "=" * 70)
    print("Perplexity Evaluation")
    print("=" * 70)
    print(f"  Model: {args.preset}")
    print(f"  Dataset: WikiText-2 (test split)")
    print(f"  Mode: {eval_mode}")
    if not args.autoregressive:
        print(f"  Stride: {args.stride}")
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Versions: {', '.join(args.version)}")
    print("=" * 70)

    # Evaluate each version
    results = {}

    for version in args.version:
        # Load model configured for this version using unified registry
        model, v_config = load_model_for_version(version, args.preset, args.dtype)
        model_config = get_model_config_dict(model)

        desc = v_config['description']
        use_cache = v_config['use_cache']
        print(f"\n--- Evaluating {version} ({desc}) ---")
        print(f"    kv_cache_quant={v_config['kv_cache_quant']}, cross_layer_sharing={v_config['cross_layer_sharing']}")

        if args.autoregressive:
            # Autoregressive: processes one token at a time with KV-cache
            # This actually tests the cache optimizations
            ppl, loss, n_tokens = compute_perplexity_autoregressive(
                model, tokens,
                use_cache=use_cache,
                cross_layer_sharing=v_config['cross_layer_sharing']
            )
        else:
            # Teacher-forcing: full sequence forward pass (cache not used)
            # All versions produce identical perplexity
            ppl, loss, n_tokens = compute_perplexity(
                model, tokens, stride=args.stride
            )

        results[version] = {
            'perplexity': ppl,
            'loss': loss,
            'num_tokens': n_tokens
        }

        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Loss: {loss:.4f}")

        # Log result
        result = {
            'benchmark_name': 'perplexity',
            'version': version,
            'version_description': desc,
            'use_cache': use_cache,
            'kv_cache_quant': v_config['kv_cache_quant'],
            'cross_layer_sharing': v_config['cross_layer_sharing'],
            'dataset': 'wikitext-2',
            'eval_mode': eval_mode,
            'stride': args.stride if not args.autoregressive else None,
            'num_tokens_evaluated': n_tokens,
            'perplexity': ppl,
            'loss': loss,
        }
        log_result(result, args.log_file, model_config, args.preset, args.dtype)

        # Clean up model
        del model
        torch.cuda.empty_cache()

    # Print summary table
    print_results_table(results, args.version)
    print(f"\nResults saved to {args.log_file}")


if __name__ == '__main__':
    main()
