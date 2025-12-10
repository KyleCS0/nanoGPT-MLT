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

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass with optional KV-cache support.

        Args:
            x: Input tensor (B, T, C)
               - Training: T = sequence length
               - Cached inference: T = 1 (new token only)
            layer_past: Optional tuple (past_key, past_value) each (B, nh, T_past, hs)
            use_cache: If True, return updated KV cache

        Returns:
            y: Output tensor (B, T, C)
            present: Tuple (key, value) if use_cache=True, else None
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # concatenate with past KV if available
        if layer_past is not None:
            if self.config.kv_cache_quant:
                # Dequantize
                (past_key_q, past_key_s), (past_value_q, past_value_s) = layer_past
                past_key = past_key_q.to(torch.float16) * past_key_s
                past_value = past_value_q.to(torch.float16) * past_value_s
            else:
                past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=2)   # (B, nh, T_past + T, hs)
            v = torch.cat([past_value, v], dim=2) # (B, nh, T_past + T, hs)

        # prepare cache to return
        if use_cache:
            if self.config.kv_cache_quant:
                # Quantize
                k_scale = k.abs().max(dim=-1, keepdim=True).values / 127.0
                k_quant = (k / k_scale).round().to(torch.int8)
                v_scale = v.abs().max(dim=-1, keepdim=True).values / 127.0
                v_quant = (v / v_scale).round().to(torch.int8)
                present = ((k_quant, k_scale), (v_quant, v_scale))
            else:
                present = (k, v)
        else:
            present = None

        # causal self-attention
        if self.flash:
            # Flash Attention: is_causal=True for no cache, False for single-token cached generation
            assert layer_past is None or T == 1, "Cached generation only supports single-token input"
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=(layer_past is None)
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
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

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

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass with optional KV-cache support.

        Args:
            x: Input tensor (B, T, C)
            layer_past: Optional KV cache for this block's attention layer
            use_cache: If True, return updated KV cache

        Returns:
            x: Output tensor (B, T, C)
            present: KV cache tuple if use_cache=True, else None
        """
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
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

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False, cross_layer_sharing=None):
        """
        Forward pass with optional KV-cache support.

        Args:
            idx: Input token indices (B, T)
            targets: Optional target indices for loss computation (B, T)
            past_key_values: Optional list of KV caches, one per layer
                            Each element is (key, value) tuple
            use_cache: If True, return updated KV caches
            cross_layer_sharing: If True, enable cross-layer sharing hack

        Returns:
            logits: Output logits (B, T, vocab_size) or (B, 1, vocab_size) if no targets
            loss: Cross-entropy loss if targets provided, else None
            present_key_values: List of KV caches if use_cache=True, else None
        """
        device = idx.device
        b, t = idx.size()

        # Determine if we are using cross-layer sharing
        use_cls = cross_layer_sharing if cross_layer_sharing is not None else self.config.cross_layer_sharing
        
        # Calculate position offset from cache
        past_length = 0
        if past_key_values is not None:
            if self.config.kv_cache_quant:
                past_length = past_key_values[0][0][0].size(2)  # (B, nh, T_past, hs)
            else:
                past_length = past_key_values[0][0].size(2)  # (B, nh, T_past, hs)

        # Validate sequence length
        total_length = past_length + t
        assert total_length <= self.config.block_size, \
            f"Cannot forward sequence of length {total_length}, block size is only {self.config.block_size}"

        # Position indices: offset by past_length when using cache
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Process through transformer blocks, collecting KV caches
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            if use_cls:
                cache_idx = i if i % 2 == 0 else i - 1 # Cross-layer sharing for inference-time hack
                layer_past = past_key_values[cache_idx] if past_key_values is not None else None
            else:
                layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                present_key_values.append(present)

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
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None, override_kv_cache_quant=None):
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
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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
        past_key_values = None

        for _ in range(max_new_tokens):
            # Determine what to feed the model
            if use_cache and past_key_values is not None:
                # Cached generation: only feed the last token
                idx_cond = idx[:, -1:]
            else:
                # First iteration or non-cached: feed full sequence (up to block_size)
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _, past_key_values = self(
                idx_cond,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                cross_layer_sharing=cross_layer_sharing
            )

            # Handle cache overflow: if we exceed block_size, trim the cache
            if use_cache and past_key_values is not None:
                if self.config.kv_cache_quant:
                    cache_len = past_key_values[0][0][0].size(2)
                else:
                    cache_len = past_key_values[0][0].size(2)
                if cache_len >= self.config.block_size:
                    # Trim cache to block_size - 1 to make room for next token
                    if self.config.kv_cache_quant:
                        past_key_values = [
                            ((k[0][:, :, -(self.config.block_size - 1):, :], k[1]),
                             (v[0][:, :, -(self.config.block_size - 1):, :], v[1]))
                            for k, v in past_key_values
                        ]
                    else:
                        past_key_values = [
                            (k[:, :, -(self.config.block_size - 1):, :],
                             v[:, :, -(self.config.block_size - 1):, :])
                            for k, v in past_key_values
                        ]

            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
