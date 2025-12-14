"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class KVCache:
    """
    Pre-allocated KV cache for efficient autoregressive generation.

    Avoids O(T²) memory allocations from torch.cat by using pre-allocated
    buffers with in-place slice assignment.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        n_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        cross_layer_sharing: bool = False,
        kv_cache_quant: bool = False,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.cross_layer_sharing = cross_layer_sharing
        self.kv_cache_quant = kv_cache_quant
        self.device = device
        self.dtype = dtype

        # Number of cache entries (half for cross-layer sharing)
        self.n_cache_layers = n_layers // 2 if cross_layer_sharing else n_layers

        # Pre-allocate buffers
        cache_shape = (self.n_cache_layers, batch_size, n_heads, max_seq_len, head_dim)

        if kv_cache_quant:
            # INT8 quantized cache
            self.k_cache = torch.empty(cache_shape, device=device, dtype=torch.int8)
            self.v_cache = torch.empty(cache_shape, device=device, dtype=torch.int8)
            # Per-token scales (one scale per position per layer)
            self.k_scale = torch.ones((self.n_cache_layers, max_seq_len), device=device, dtype=dtype)
            self.v_scale = torch.ones((self.n_cache_layers, max_seq_len), device=device, dtype=dtype)
        else:
            # FP16/BF16 cache
            self.k_cache = torch.empty(cache_shape, device=device, dtype=dtype)
            self.v_cache = torch.empty(cache_shape, device=device, dtype=dtype)
            self.k_scale = None
            self.v_scale = None

        self.seq_len = 0  # Current filled length

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               k_scale: torch.Tensor = None, v_scale: torch.Tensor = None):
        """
        Update cache for a layer with new K, V values.

        Args:
            layer_idx: Cache layer index (not model layer index)
            k: New key tensor (B, nh, T_new, hs) - quantized if kv_cache_quant
            v: New value tensor (B, nh, T_new, hs) - quantized if kv_cache_quant
            k_scale, v_scale: Scales for INT8 quantization

        Returns:
            T_new: Number of new tokens added (for get() to include them)
        """
        T_new = k.size(2)
        start_pos = self.seq_len
        end_pos = start_pos + T_new

        # In-place assignment (no allocation!)
        self.k_cache[layer_idx, :, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx, :, :, start_pos:end_pos, :] = v

        if self.kv_cache_quant and k_scale is not None:
            # Store per-token scales (not per-layer!)
            self.k_scale[layer_idx, start_pos:end_pos] = k_scale
            self.v_scale[layer_idx, start_pos:end_pos] = v_scale

        return T_new

    def get(self, layer_idx: int, include_new: int = 0):
        """
        Get cached K, V for a layer.

        Args:
            layer_idx: Cache layer index
            include_new: Include this many tokens beyond seq_len (for just-written tokens)

        Returns:
            For FP16: (k, v) each of shape (B, nh, seq_len + include_new, hs)
            For INT8: ((k_quant, k_scale), (v_quant, v_scale))
        """
        end_pos = self.seq_len + include_new
        k = self.k_cache[layer_idx, :, :, :end_pos, :]
        v = self.v_cache[layer_idx, :, :, :end_pos, :]

        if self.kv_cache_quant:
            # Return per-token scales for correct dequantization
            return ((k, self.k_scale[layer_idx, :end_pos]), (v, self.v_scale[layer_idx, :end_pos]))
        else:
            return (k, v)

    def advance(self, n_tokens: int = 1):
        """Advance the sequence position after processing tokens."""
        self.seq_len += n_tokens

    def trim(self, keep_len: int):
        """
        Trim cache to keep only the last keep_len tokens (for sliding window).
        Uses in-place copy to avoid allocation.
        """
        if self.seq_len <= keep_len:
            return

        # Shift the last keep_len tokens to the beginning
        start = self.seq_len - keep_len
        self.k_cache[:, :, :, :keep_len, :] = self.k_cache[:, :, :, start:self.seq_len, :].clone()
        self.v_cache[:, :, :, :keep_len, :] = self.v_cache[:, :, :, start:self.seq_len, :].clone()
        # Also shift per-token scales if INT8 quantized
        if self.kv_cache_quant:
            self.k_scale[:, :keep_len] = self.k_scale[:, start:self.seq_len].clone()
            self.v_scale[:, :keep_len] = self.v_scale[:, start:self.seq_len].clone()
        self.seq_len = keep_len

    def reset(self):
        """Reset cache for new sequence."""
        self.seq_len = 0


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None, use_cache=False, is_cache_owner=True,
                kv_cache: 'KVCache' = None, cache_layer_idx: int = None):
        """
        Forward pass with optional KV-cache support.

        Args:
            x: Input tensor (B, T, C)
               - Training: T = sequence length
               - Cached inference: T = 1 (new token only)
            layer_past: Optional tuple (past_key, past_value) each (B, nh, T_past, hs)
                       [DEPRECATED: use kv_cache instead for better performance]
            use_cache: If True, return updated KV cache
            is_cache_owner: If True, this layer is responsible for quantizing the cache
            kv_cache: Optional KVCache object for pre-allocated cache (avoids torch.cat)
            cache_layer_idx: Layer index in the cache (required if kv_cache is provided)

        Returns:
            y: Output tensor (B, T, C)
            present: Tuple (key, value) if use_cache=True and no kv_cache, else None
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Determine if we're using the optimized KVCache path
        use_kv_cache = kv_cache is not None and cache_layer_idx is not None

        # In cross-layer sharing, borrower layers (is_cache_owner=False) with a valid cache
        # should only compute Q and reuse K,V from the cache owner.
        # Note: with KVCache, owner writes before borrower reads (same forward pass),
        # so borrower can always borrow via include_new=T, even when seq_len=0.
        has_past = layer_past is not None or use_kv_cache
        can_borrow = self.config.cross_layer_sharing and not is_cache_owner and has_past

        if can_borrow:
            # Borrower layer: compute Q only, K and V will be taken from cache
            q = self.c_attn_q(x)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            k, v = None, None
        else:
            # Owner layer or borrower without cache: compute Q, K, V
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Handle KV cache - optimized path using pre-allocated KVCache
        if use_kv_cache:
            if can_borrow:
                # Borrower: just get K,V from cache (no update needed)
                # include_new=T because owner has written new tokens but advance() not called yet
                layer_cache = kv_cache.get(cache_layer_idx, include_new=T)
                if self.config.kv_cache_quant:
                    (k_quant, k_scale), (v_quant, v_scale) = layer_cache
                    # Per-token dequantization: scale is (seq_len,), broadcast to (1, 1, seq_len, 1)
                    k = k_quant.to(q.dtype) * k_scale.view(1, 1, -1, 1)
                    v = v_quant.to(q.dtype) * v_scale.view(1, 1, -1, 1)
                else:
                    k, v = layer_cache
            else:
                # Owner: update cache in-place, then get full K,V
                if self.config.kv_cache_quant:
                    # Quantize new K,V
                    k_absmax = k.abs().max()
                    k_scale = k_absmax / 127.0 if k_absmax > 0 else torch.tensor(1.0, device=k.device, dtype=k.dtype)
                    k_quant = (k / k_scale).round().clamp(-127, 127).to(torch.int8)
                    v_absmax = v.abs().max()
                    v_scale = v_absmax / 127.0 if v_absmax > 0 else torch.tensor(1.0, device=v.device, dtype=v.dtype)
                    v_quant = (v / v_scale).round().clamp(-127, 127).to(torch.int8)
                    # Update cache in-place
                    kv_cache.update(cache_layer_idx, k_quant, v_quant, k_scale, v_scale)
                else:
                    # Update cache in-place (no quantization)
                    kv_cache.update(cache_layer_idx, k, v)

                # Get full K,V from cache (past + new, as a view - no allocation!)
                # include_new=T because we just wrote T tokens but advance() not called yet
                layer_cache = kv_cache.get(cache_layer_idx, include_new=T)
                if self.config.kv_cache_quant:
                    (k_quant, k_scale), (v_quant, v_scale) = layer_cache
                    # Per-token dequantization: scale is (seq_len,), broadcast to (1, 1, seq_len, 1)
                    k = k_quant.to(q.dtype) * k_scale.view(1, 1, -1, 1)
                    v = v_quant.to(q.dtype) * v_scale.view(1, 1, -1, 1)
                else:
                    k, v = layer_cache

            present = None  # With KVCache, we don't return presents

        # Legacy path using layer_past (torch.cat, kept for backward compatibility)
        elif layer_past is not None:
            if self.config.kv_cache_quant:
                # INT8 path: dequantize cache, concat with new K,V
                (past_key_q, past_key_s), (past_value_q, past_value_s) = layer_past
                past_key = past_key_q.to(q.dtype) * past_key_s
                past_value = past_value_q.to(q.dtype) * past_value_s
                if can_borrow:
                    k, v = past_key, past_value
                else:
                    k = torch.cat([past_key, k], dim=2)
                    v = torch.cat([past_value, v], dim=2)
            else:
                past_key, past_value = layer_past
                if can_borrow:
                    k, v = past_key, past_value
                else:
                    k = torch.cat([past_key, k], dim=2)
                    v = torch.cat([past_value, v], dim=2)

        # prepare cache to return (only for legacy path, KVCache handles its own updates)
        if not use_kv_cache:
            if use_cache:
                if self.config.kv_cache_quant and is_cache_owner:
                    # Quantize for storage (saves memory, costs some compute)
                    k_absmax = k.abs().max()
                    k_scale = k_absmax / 127.0 if k_absmax > 0 else torch.tensor(1.0, device=k.device, dtype=k.dtype)
                    k_quant = (k / k_scale).round().clamp(-127, 127).to(torch.int8)
                    v_absmax = v.abs().max()
                    v_scale = v_absmax / 127.0 if v_absmax > 0 else torch.tensor(1.0, device=v.device, dtype=v.dtype)
                    v_quant = (v / v_scale).round().clamp(-127, 127).to(torch.int8)
                    present = ((k_quant, k_scale), (v_quant, v_scale))
                else:
                    present = (k, v)
            else:
                present = None

        # causal self-attention
        if self.flash:
            # Flash Attention: is_causal=True for no cache, False for single-token cached generation
            has_cache = layer_past is not None or (use_kv_cache and kv_cache.seq_len > T)
            assert not has_cache or T == 1, "Cached generation only supports single-token input"
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=(not has_cache)
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            T_total = k.size(2)
            T_past = T_total - T
            att = att.masked_fill(self.bias[:, :, T_past:T_total, :T_total] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None, use_cache=False, is_cache_owner=True,
                kv_cache: 'KVCache' = None, cache_layer_idx: int = None):
        """
        Forward pass with optional KV-cache support.

        Args:
            x: Input tensor (B, T, C)
            layer_past: Optional KV cache for this block's attention layer
            use_cache: If True, return updated KV cache
            is_cache_owner: If True, this layer is responsible for quantizing the cache
            kv_cache: Optional KVCache object for pre-allocated cache
            cache_layer_idx: Layer index in the cache

        Returns:
            x: Output tensor (B, T, C)
            present: KV cache tuple if use_cache=True and no kv_cache, else None
        """
        attn_out, present = self.attn(
            self.ln_1(x), layer_past=layer_past, use_cache=use_cache,
            is_cache_owner=is_cache_owner, kv_cache=kv_cache, cache_layer_idx=cache_layer_idx
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    cross_layer_sharing: bool = False # Toggles the cross-layer sharing hack
    kv_cache_quant: bool = False # Toggles INT8 quantization for KV cache
    cross_layer_q_alignment: bool = False # Align borrower Q weights with owner (experimental)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Pre-compute position indices for efficient inference
        self.register_buffer('position_ids', torch.arange(config.block_size, dtype=torch.long))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False,
                cross_layer_sharing=None, kv_cache: 'KVCache' = None):
        """
        Forward pass with optional KV-cache support.

        Args:
            idx: Input token indices (B, T)
            targets: Optional target indices for loss computation (B, T)
            past_key_values: Optional list of KV caches, one per layer
                            Each element is (key, value) tuple
                            [DEPRECATED: use kv_cache for better performance]
            use_cache: If True, return updated KV caches (ignored if kv_cache provided)
            cross_layer_sharing: If True, enable cross-layer sharing hack
            kv_cache: Optional KVCache object for pre-allocated cache (avoids torch.cat)

        Returns:
            logits: Output logits (B, T, vocab_size) or (B, 1, vocab_size) if no targets
            loss: Cross-entropy loss if targets provided, else None
            present_key_values: List of KV caches if use_cache=True and no kv_cache, else None
        """
        device = idx.device
        b, t = idx.size()

        # Determine if we are using cross-layer sharing
        use_cls = cross_layer_sharing if cross_layer_sharing is not None else self.config.cross_layer_sharing

        # Determine if using optimized KVCache path
        use_kv_cache = kv_cache is not None

        # Calculate position offset from cache
        past_length = 0
        if use_kv_cache:
            past_length = kv_cache.seq_len
        elif past_key_values is not None:
            if self.config.kv_cache_quant:
                past_length = past_key_values[0][0][0].size(2)  # (B, nh, T_past, hs)
            else:
                past_length = past_key_values[0][0].size(2)  # (B, nh, T_past, hs)

        # Validate sequence length
        total_length = past_length + t
        assert total_length <= self.config.block_size, \
            f"Cannot forward sequence of length {total_length}, block size is only {self.config.block_size}"

        # Position indices: use pre-computed buffer, slice for current positions
        pos = self.position_ids[past_length:past_length + t]

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Process through transformer blocks
        present_key_values = [] if (use_cache and not use_kv_cache) else None
        for i, block in enumerate(self.transformer.h):
            if use_cls:
                cache_idx = i // 2 # e.g. layer 0,1 -> 0; layer 2,3 -> 1
                is_cache_owner = (i % 2 == 0)
                layer_past = past_key_values[cache_idx] if past_key_values is not None else None
            else:
                cache_idx = i
                is_cache_owner = True
                layer_past = past_key_values[i] if past_key_values is not None else None

            # Pass KVCache if using optimized path
            # Note: cache_layer_idx must be passed to BOTH owner and borrower layers
            x, present = block(
                x, layer_past=layer_past, use_cache=use_cache, is_cache_owner=is_cache_owner,
                kv_cache=kv_cache if use_kv_cache else None,
                cache_layer_idx=cache_idx if use_kv_cache else None
            )

            if use_cache and not use_kv_cache:
                # with cross-layer sharing, only even layers store their cache
                if not use_cls or is_cache_owner:
                    present_key_values.append(present)

        # Advance KVCache position after all layers have updated it
        if use_kv_cache:
            kv_cache.advance(t)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, present_key_values

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        self.position_ids = self.position_ids[:block_size]
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None, override_kv_cache_quant=None,
                        override_cross_layer_q_alignment=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # override kv_cache_quant if desired
        if override_kv_cache_quant is not None:
            print(f"overriding kv_cache_quant to {override_kv_cache_quant}")
            config_args['kv_cache_quant'] = override_kv_cache_quant
        # override cross_layer_q_alignment if desired (for experimental versions)
        if override_cross_layer_q_alignment is not None:
            print(f"overriding cross_layer_q_alignment to {override_cross_layer_q_alignment}")
            config_args['cross_layer_q_alignment'] = override_cross_layer_q_alignment
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # Filter out c_attn_q (our addition for cross-layer sharing) and position_ids (our registered buffer)
        sd_keys_ours = [k for k in sd_keys if 'c_attn_q' not in k and k != 'position_ids']
        assert len(sd_keys_hf) == len(sd_keys_ours), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys_ours)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        # Migrate weights for the new c_attn_q layer
        # If cross_layer_q_alignment is enabled, borrower layers (odd indices) use the
        # OWNER's Q weights so Q and K come from the same learned weight space.
        use_q_alignment = getattr(config, 'cross_layer_q_alignment', False)
        with torch.no_grad():
            for i, h in enumerate(model.transformer.h):
                if use_q_alignment and i % 2 == 1:
                    # Borrower layer with Q-alignment: use owner's Q weights
                    owner = model.transformer.h[i - 1].attn
                    q_weight = owner.c_attn.weight[:config.n_embd, :]
                    q_bias = owner.c_attn.bias[:config.n_embd]
                else:
                    # Owner layer or no Q-alignment: use own Q weights
                    q_weight = h.attn.c_attn.weight[:config.n_embd, :]
                    q_bias = h.attn.c_attn.bias[:config.n_embd]
                h.attn.c_attn_q.weight.copy_(q_weight)
                h.attn.c_attn_q.bias.copy_(q_bias)

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True, cross_layer_sharing=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens
            use_cache: If True, use KV-cache for efficient generation (default: True)
            cross_layer_sharing: If True, enable cross-layer sharing hack

        Returns:
            idx: Generated sequence (B, T + max_new_tokens)
        """
        B, T = idx.size()
        device = idx.device

        # Pre-allocate output buffer to avoid torch.cat every iteration
        output = torch.empty(B, T + max_new_tokens, dtype=idx.dtype, device=device)
        output[:, :T] = idx
        cur_pos = T  # Current write position

        # Determine cross-layer sharing setting
        use_cls = cross_layer_sharing if cross_layer_sharing is not None else self.config.cross_layer_sharing

        # Create pre-allocated KVCache for optimized generation
        kv_cache = None
        if use_cache:
            # Get model dtype from parameters
            dtype = next(self.parameters()).dtype
            kv_cache = KVCache(
                batch_size=B,
                max_seq_len=self.config.block_size,
                n_heads=self.config.n_head,
                head_dim=self.config.n_embd // self.config.n_head,
                n_layers=self.config.n_layer,
                device=device,
                dtype=dtype,
                cross_layer_sharing=use_cls,
                kv_cache_quant=self.config.kv_cache_quant,
            )

        for i in range(max_new_tokens):
            # Determine what to feed the model
            if use_cache and kv_cache.seq_len > 0:
                # Cached generation: only feed the last token
                idx_cond = output[:, cur_pos - 1:cur_pos]
            else:
                # First iteration or non-cached: feed full sequence (up to block_size)
                idx_cond = output[:, :cur_pos]
                if idx_cond.size(1) > self.config.block_size:
                    idx_cond = idx_cond[:, -self.config.block_size:]

            # Forward pass with pre-allocated KVCache
            logits, _, _ = self(
                idx_cond,
                use_cache=use_cache,
                cross_layer_sharing=cross_layer_sharing,
                kv_cache=kv_cache
            )

            # Handle cache overflow: if we exceed block_size, trim the cache
            if use_cache and kv_cache.seq_len >= self.config.block_size:
                # Trim cache to block_size - 1 to make room for next token
                kv_cache.trim(self.config.block_size - 1)

            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            if top_k is not None:
                # Optimized top-k: only compute softmax on top-k values
                k = min(top_k, logits.size(-1))
                topk_logits, topk_indices = torch.topk(logits, k)
                topk_probs = F.softmax(topk_logits, dim=-1)
                sampled_idx = torch.multinomial(topk_probs, num_samples=1)
                idx_next = torch.gather(topk_indices, dim=-1, index=sampled_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # Write to pre-allocated buffer (no allocation)
            output[:, cur_pos] = idx_next.squeeze(-1)
            cur_pos += 1

        return output
