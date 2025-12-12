# benchmark/versions.py
"""
Version registry for KV-cache optimization benchmarks.

This is the SINGLE SOURCE OF TRUTH for version definitions.
All benchmark scripts should import from here.

Versions:
- v0: No cache (baseline) - recomputes full sequence each step
- v1: KV-cache enabled - caches key/value pairs for efficiency
- v2: KV-cache + INT8 quantization - reduces cache memory via quantization
- v3: KV-cache + cross-layer sharing - shares KV across layer pairs
- v4: KV-cache + INT8 + cross-layer sharing - maximum memory optimization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig

VERSIONS = {
    'v0': {
        'description': 'No cache (baseline)',
        'use_cache': False,
        'kv_cache_quant': False,
        'cross_layer_sharing': False,
    },
    'v1': {
        'description': 'KV-cache enabled',
        'use_cache': True,
        'kv_cache_quant': False,
        'cross_layer_sharing': False,
    },
    'v2': {
        'description': 'KV-cache + INT8 quantization',
        'use_cache': True,
        'kv_cache_quant': True,
        'cross_layer_sharing': False,
    },
    'v3': {
        'description': 'KV-cache + cross-layer sharing',
        'use_cache': True,
        'kv_cache_quant': False,
        'cross_layer_sharing': True,
    },
    'v4': {
        'description': 'KV-cache + INT8 + cross-layer sharing',
        'use_cache': True,
        'kv_cache_quant': True,
        'cross_layer_sharing': True,
    },
}


def get_version_config(version: str) -> dict:
    """
    Get the full configuration for a version.

    Args:
        version: Version string (v0, v1, v2, v3, v4)

    Returns:
        dict with keys: description, use_cache, kv_cache_quant, cross_layer_sharing
    """
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")
    return VERSIONS[version].copy()


def get_use_cache(version: str) -> bool:
    """Get the use_cache setting for a version."""
    return get_version_config(version)['use_cache']


def get_kv_cache_quant(version: str) -> bool:
    """Get the kv_cache_quant setting for a version."""
    return get_version_config(version)['kv_cache_quant']


def get_cross_layer_sharing(version: str) -> bool:
    """Get the cross_layer_sharing setting for a version."""
    return get_version_config(version)['cross_layer_sharing']


def create_model(version: str, base_config: dict = None, pretrained: str = None):
    """
    Create model for specified version.

    Args:
        version: Version string (v0, v1, v2, v3, v4)
        base_config: Base model configuration dict (used if pretrained is None)
        pretrained: Pretrained model name (gpt2, gpt2-medium, etc.)

    Returns:
        model: GPT model instance
        version_config: Version configuration dict
    """
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")

    v_config = VERSIONS[version]

    if pretrained:
        # Load pretrained with version-specific settings
        model = GPT.from_pretrained(
            pretrained,
            override_kv_cache_quant=v_config['kv_cache_quant']
        )
        # Set cross_layer_sharing in config (used at runtime)
        model.config.cross_layer_sharing = v_config['cross_layer_sharing']
    else:
        # Create from scratch with version-specific settings
        base_config = base_config or {}
        config = GPTConfig(
            **base_config,
            kv_cache_quant=v_config['kv_cache_quant'],
            cross_layer_sharing=v_config['cross_layer_sharing']
        )
        model = GPT(config)

    return model, v_config


def load_model_for_version(version: str, pretrained: str, dtype_str: str = 'bfloat16'):
    """
    Load a model configured for a specific version.

    This is the recommended way to load models for benchmarking.

    Args:
        version: Version string (v0, v1, v2, v3, v4)
        pretrained: Pretrained model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        dtype_str: Data type string ('float32', 'bfloat16', 'float16')

    Returns:
        model: GPT model on CUDA with specified dtype
        version_config: Version configuration dict
    """
    import torch

    model, v_config = create_model(version, pretrained=pretrained)
    model.to("cuda")

    pt_dtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[dtype_str]
    model.to(pt_dtype)
    model.eval()

    return model, v_config


def get_model_config_dict(model):
    """Extract model configuration as a dict for logging."""
    return {
        'n_layer': model.config.n_layer,
        'n_head': model.config.n_head,
        'n_embd': model.config.n_embd,
        'block_size': model.config.block_size,
        'vocab_size': model.config.vocab_size,
        'dropout': model.config.dropout,
        'bias': model.config.bias,
        'kv_cache_quant': model.config.kv_cache_quant,
        'cross_layer_sharing': model.config.cross_layer_sharing,
    }


def list_versions():
    """Print all available versions."""
    print("Available versions:")
    for name, info in VERSIONS.items():
        print(f"  {name}: {info['description']}")
        print(f"       use_cache: {info['use_cache']}")
        print(f"       kv_cache_quant: {info['kv_cache_quant']}")
        print(f"       cross_layer_sharing: {info['cross_layer_sharing']}")


if __name__ == '__main__':
    list_versions()
