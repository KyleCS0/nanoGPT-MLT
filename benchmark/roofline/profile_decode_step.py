"""
Roofline profiling script for KV Cache analysis.

Multi-stage profiling to evaluate different KV cache optimization variants:

Base versions (prefill/decode):
  v0_prefill: Full forward on P tokens, no cache (baseline)
  v1_prefill: Full forward on P tokens, with cache enabled (cache build)
  v1_decode:  Single token decode using cached K/V (cache reuse)

Extended versions (for v2/v3/v4):
  v2_prefill: Full forward with INT8 quantized cache
  v2_decode:  Single token decode with INT8 quantized cache
  v3_prefill: Full forward with cross-layer sharing
  v3_decode:  Single token decode with cross-layer sharing
  v4_prefill: Full forward with INT8 + cross-layer sharing
  v4_decode:  Single token decode with INT8 + cross-layer sharing

Usage:
    python profile_decode_step.py --version v0_prefill --P 512
    python profile_decode_step.py --version v1_decode --P 512
    python profile_decode_step.py --version v2_decode --P 512  # INT8 quantization
    python profile_decode_step.py --version v3_decode --P 512  # Cross-layer sharing
    python profile_decode_step.py --version v4_decode --P 512  # INT8 + Cross-layer
"""
import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model import GPT
from benchmark.versions import VERSIONS, load_model_for_version, get_version_config

# Map roofline version names to base version and phase
# e.g., 'v2_decode' -> version='v2', phase='decode'
ROOFLINE_VERSIONS = {}
for v in VERSIONS.keys():
    ROOFLINE_VERSIONS[f'{v}_prefill'] = {'version': v, 'phase': 'prefill'}
    if v != 'v0':  # v0 has no decode (no cache)
        ROOFLINE_VERSIONS[f'{v}_decode'] = {'version': v, 'phase': 'decode'}

# Fixed prompt tokens - excerpt from "Attention Is All You Need" abstract
# This provides reproducible, meaningful input rather than random tokens
PROMPT_TEXT = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves a new state of the art on English-to-German translation, improving over the existing best results. On the English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score, after training for less than half the cost of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."""


def get_prompt_tokens(tokenizer_encode, P, device, batch_size):
    """Get P prompt tokens, repeating text if needed."""
    # Encode the fixed prompt
    tokens = tokenizer_encode(PROMPT_TEXT)

    # Repeat if we need more tokens than available
    while len(tokens) < P:
        tokens = tokens + tokens

    # Truncate to exactly P tokens
    tokens = tokens[:P]

    # Create batch
    idx = torch.tensor(tokens, device=device).unsqueeze(0).expand(batch_size, -1)
    return idx


def profile_prefill(model, prompt_tokens, device, use_cache=False, cross_layer_sharing=None):
    """Prefill: Full forward pass on P tokens.

    Args:
        model: GPT model
        prompt_tokens: Input tokens (B, P)
        device: CUDA device
        use_cache: Whether to build KV cache
        cross_layer_sharing: Override for cross-layer sharing setting

    Returns:
        cache: KV cache if use_cache=True, else None
    """
    torch.cuda.synchronize()
    with torch.no_grad():
        logits, _, cache = model(prompt_tokens, use_cache=use_cache,
                                 cross_layer_sharing=cross_layer_sharing)
    torch.cuda.synchronize()
    return cache


def profile_decode(model, cache, device, batch_size, cross_layer_sharing=None):
    """Decode: Single token forward using cached K/V.

    This is the key comparison point:
    - Processes only 1 new token
    - Reads cached K/V (P entries per layer)
    - Should show dramatically lower FLOPs
    - Should be memory-bound (low arithmetic intensity)

    Args:
        model: GPT model
        cache: KV cache from prefill
        device: CUDA device
        batch_size: Batch size
        cross_layer_sharing: Override for cross-layer sharing setting
    """
    # Single new token (position P+1)
    new_token = torch.randint(0, 50257, (batch_size, 1), device=device)

    torch.cuda.synchronize()
    with torch.no_grad():
        logits, _, _ = model(new_token, past_key_values=cache, use_cache=True,
                            cross_layer_sharing=cross_layer_sharing)
    torch.cuda.synchronize()

def main():
    # Build choices from ROOFLINE_VERSIONS
    available_versions = list(ROOFLINE_VERSIONS.keys())
    version_help = "Available versions:\n"
    for rv, info in ROOFLINE_VERSIONS.items():
        base_v = info['version']
        phase = info['phase']
        base_config = get_version_config(base_v)
        version_help += f"  {rv}: {base_config['description']} ({phase})\n"

    parser = argparse.ArgumentParser(
        description='Roofline profiling for KV Cache analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=version_help
    )
    parser.add_argument('--version', type=str, required=True,
                        choices=available_versions,
                        help='Profiling version (see below for full list)')
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Model: gpt2, gpt2-medium, gpt2-large, gpt2-xl')
    parser.add_argument('--P', type=int, default=512,
                        help='Prompt length (number of tokens for prefill)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    args = parser.parse_args()

    device = 'cuda'

    # Parse roofline version to get base version and phase
    rv_info = ROOFLINE_VERSIONS[args.version]
    base_version = rv_info['version']
    phase = rv_info['phase']
    v_config = get_version_config(base_version)

    use_cache = v_config['use_cache']
    cross_layer_sharing = v_config['cross_layer_sharing']

    # Load model configured for this version
    print(f"Loading {args.model} for {args.version}...")
    print(f"  Base version: {base_version} ({v_config['description']})")
    print(f"  Phase: {phase}")
    print(f"  kv_cache_quant: {v_config['kv_cache_quant']}")
    print(f"  cross_layer_sharing: {cross_layer_sharing}")

    model, _ = load_model_for_version(base_version, args.model, args.dtype)

    # Get tokenizer for encoding prompt
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Get prompt tokens
    prompt_tokens = get_prompt_tokens(enc.encode, args.P, device, args.batch_size)
    print(f"\nPrompt: {args.P} tokens, Batch: {args.batch_size}")

    # Warmup (use same code path as profiled version)
    warmup_tokens = prompt_tokens[:, :min(32, args.P)]
    with torch.no_grad():
        for _ in range(3):
            if phase == 'prefill' and not use_cache:
                model(warmup_tokens, use_cache=False)
            else:
                _, _, warmup_cache = model(warmup_tokens, use_cache=True,
                                           cross_layer_sharing=cross_layer_sharing)
                if phase == 'decode':
                    # Also warmup decode path
                    single_tok = torch.randint(0, 50257, (args.batch_size, 1), device=device)
                    model(single_tok, past_key_values=warmup_cache, use_cache=True,
                          cross_layer_sharing=cross_layer_sharing)
    torch.cuda.synchronize()

    # Profile based on phase
    if phase == 'prefill':
        print(f"\nProfiling {args.version}: {args.P} tokens, use_cache={use_cache}...")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_prefill(model, prompt_tokens, device, use_cache=use_cache,
                       cross_layer_sharing=cross_layer_sharing)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()

    elif phase == 'decode':
        # Build cache first (not profiled), then profile single token decode
        print(f"\nBuilding cache from {args.P} tokens (not profiled)...")
        with torch.no_grad():
            _, _, cache = model(prompt_tokens, use_cache=True,
                               cross_layer_sharing=cross_layer_sharing)
        torch.cuda.synchronize()

        print(f"Profiling {args.version}: 1 token with cache of {args.P} entries...")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_decode(model, cache, device, args.batch_size,
                      cross_layer_sharing=cross_layer_sharing)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()

    print("\nProfiling complete.")


if __name__ == '__main__':
    main()
