"""
Decode step profiling script for Nsight Compute.
Profiles a single forward pass at sequence length T.

Usage:
    python profile_decode_step.py --version v0 --T 128
    python profile_decode_step.py --version v1 --T 128
"""
import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model import GPT

def profile_v0(model, T, batch_size, device):
    """V0: Full sequence forward pass (no cache)."""
    # Full sequence of T tokens
    idx = torch.randint(0, 50257, (batch_size, T), device=device)

    torch.cuda.synchronize()
    # torch.cuda.profiler.start()
    with torch.no_grad():
        logits, _, _ = model(idx, use_cache=False)
    # torch.cuda.profiler.stop()
    torch.cuda.synchronize()

def profile_v1(model, T, batch_size, device):
    """V1: Single token forward with pre-built cache."""
    # Build cache from T-1 tokens (prefill) - Not profiled
    prompt = torch.randint(0, 50257, (batch_size, T-1), device=device)
    with torch.no_grad():
        _, _, cache = model(prompt, use_cache=True)

    torch.cuda.synchronize()

    # Profile: single token decode with cache
    new_token = torch.randint(0, 50257, (batch_size, 1), device=device)
    # torch.cuda.profiler.start()
    with torch.no_grad():
        logits, _, _ = model(new_token, past_key_values=cache, use_cache=True)
    # torch.cuda.profiler.stop()
    torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, choices=['v0', 'v1'])
    parser.add_argument('--model', type=str, default='gpt2', help='Model type: gpt2, gpt2-medium, gpt2-large, gpt2-xl')
    parser.add_argument('--T', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    args = parser.parse_args()

    device = 'cuda'
    dtype = getattr(torch, args.dtype)

    # Load model
    print(f"Loading {args.model} (Batch Size={args.batch_size}, T={args.T})...")
    model = GPT.from_pretrained(args.model).to(device).to(dtype).eval()

    # Warmup
    warmup_idx = torch.randint(0, 50257, (args.batch_size, 32), device=device)
    with torch.no_grad():
        for _ in range(3):
            model(warmup_idx, use_cache=False)
    torch.cuda.synchronize()

    # Profile
    if args.version == 'v0':
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_v0(model, args.T, args.batch_size, device)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        profile_v1(model, args.T, args.batch_size, device)
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()

if __name__ == '__main__':
    main()
