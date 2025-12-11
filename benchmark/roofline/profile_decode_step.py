"""
Roofline profiling script for KV Cache analysis.

Three-stage profiling to evaluate KV cache effect:
  v0_prefill: Full forward on P tokens, no cache (baseline)
  v1_prefill: Full forward on P tokens, with cache enabled (cache build)
  v1_decode:  Single token decode using cached K/V (cache reuse)

Usage:
    python profile_decode_step.py --version v0_prefill --P 512
    python profile_decode_step.py --version v1_prefill --P 512
    python profile_decode_step.py --version v1_decode --P 512
"""
import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model import GPT

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


def profile_v0_prefill(model, prompt_tokens, device):
    """V0: Full forward pass on P tokens WITHOUT KV cache.

    Baseline: Full recomputation, no caching overhead.
    This measures the cost of processing P tokens from scratch.
    """
    torch.cuda.synchronize()
    with torch.no_grad():
        logits, _, _ = model(prompt_tokens, use_cache=False)
    torch.cuda.synchronize()


def profile_v1_prefill(model, prompt_tokens, device):
    """V1 Prefill: Full forward pass on P tokens WITH KV cache enabled.

    Similar compute to V0, but also writes K/V to cache.
    Returns the cache for use in decode step.
    """
    torch.cuda.synchronize()
    with torch.no_grad():
        logits, _, cache = model(prompt_tokens, use_cache=True)
    torch.cuda.synchronize()
    return cache


def profile_v1_decode(model, cache, device, batch_size):
    """V1 Decode: Single token forward using cached K/V.

    This is the key comparison point:
    - Processes only 1 new token
    - Reads cached K/V (P entries per layer)
    - Should show dramatically lower FLOPs
    - Should be memory-bound (low arithmetic intensity)
    """
    # Single new token (position P+1)
    new_token = torch.randint(0, 50257, (batch_size, 1), device=device)

    torch.cuda.synchronize()
    with torch.no_grad():
        logits, _, _ = model(new_token, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser(description='Roofline profiling for KV Cache analysis')
    parser.add_argument('--version', type=str, required=True,
                        choices=['v0_prefill', 'v1_prefill', 'v1_decode'],
                        help='v0_prefill: no cache, v1_prefill: build cache, v1_decode: use cache')
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        help='Model: gpt2, gpt2-medium, gpt2-large, gpt2-xl')
    parser.add_argument('--P', type=int, default=512,
                        help='Prompt length (number of tokens for prefill)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    args = parser.parse_args()

    device = 'cuda'
    dtype = getattr(torch, args.dtype)

    # Load model
    print(f"Loading {args.model}...")
    model = GPT.from_pretrained(args.model).to(device).to(dtype).eval()

    # Get tokenizer for encoding prompt
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Get prompt tokens
    prompt_tokens = get_prompt_tokens(enc.encode, args.P, device, args.batch_size)
    print(f"Prompt: {args.P} tokens, Batch: {args.batch_size}, Version: {args.version}")

    # Warmup (use same code path as profiled version)
    warmup_tokens = prompt_tokens[:, :min(32, args.P)]
    with torch.no_grad():
        for _ in range(3):
            if args.version == 'v0_prefill':
                model(warmup_tokens, use_cache=False)
            else:
                _, _, warmup_cache = model(warmup_tokens, use_cache=True)
                if args.version == 'v1_decode':
                    # Also warmup decode path
                    single_tok = torch.randint(0, 50257, (args.batch_size, 1), device=device)
                    model(single_tok, past_key_values=warmup_cache, use_cache=True)
    torch.cuda.synchronize()

    # Profile based on version
    if args.version == 'v0_prefill':
        # V0: Full forward, no cache
        print(f"Profiling V0 Prefill: {args.P} tokens, no cache...")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_v0_prefill(model, prompt_tokens, device)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()

    elif args.version == 'v1_prefill':
        # V1 Prefill: Full forward, builds cache
        print(f"Profiling V1 Prefill: {args.P} tokens, building cache...")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_v1_prefill(model, prompt_tokens, device)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()

    elif args.version == 'v1_decode':
        # V1 Decode: Build cache first (not profiled), then profile single token decode
        print(f"Building cache from {args.P} tokens (not profiled)...")
        with torch.no_grad():
            _, _, cache = model(prompt_tokens, use_cache=True)
        torch.cuda.synchronize()

        print(f"Profiling V1 Decode: 1 token with cache of {args.P} entries...")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_v1_decode(model, cache, device, args.batch_size)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()


if __name__ == '__main__':
    main()
