# benchmark/versions.py
"""
Version registry for KV-cache optimization benchmarks.

Versions:
- v0: No cache (baseline) - recomputes full sequence each step
- v1: KV-cache enabled - caches key/value pairs for efficiency
- v2: KV-cache + INT8 quantization (future)
- v3: KV-cache + layer sharing (future)
- v4: KV-cache + INT8 + layer sharing (future)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig

VERSIONS = {
    'v0': {
        'description': 'No cache (baseline)',
        'config': {},
        'use_cache': False,
    },
    'v1': {
        'description': 'KV-cache enabled',
        'config': {},
        'use_cache': True,
    },
    # Future versions (Task 4, 5, 7):
    # 'v2': {
    #     'description': 'KV-cache + INT8 quantization',
    #     'config': {'quantize_cache': True},
    #     'use_cache': True,
    # },
    # 'v3': {
    #     'description': 'KV-cache + layer sharing',
    #     'config': {'share_kv_layers': 2},
    #     'use_cache': True,
    # },
    # 'v4': {
    #     'description': 'KV-cache + INT8 + layer sharing',
    #     'config': {'quantize_cache': True, 'share_kv_layers': 2},
    #     'use_cache': True,
    # },
}


def create_model(version: str, base_config: dict = None):
    """
    Create model for specified version.

    Args:
        version: Version string (v0, v1, v2, v3, v4)
        base_config: Base model configuration dict

    Returns:
        model: GPT model instance
        use_cache: Whether to use KV-cache for this version
    """
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")

    base_config = base_config or {}
    overrides = VERSIONS[version]['config']
    config = GPTConfig(**{**base_config, **overrides})
    return GPT(config), VERSIONS[version]['use_cache']


def get_use_cache(version: str) -> bool:
    """Get the use_cache setting for a version."""
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")
    return VERSIONS[version]['use_cache']


def list_versions():
    """Print all available versions."""
    print("Available versions:")
    for name, info in VERSIONS.items():
        print(f"  {name}: {info['description']}")
        print(f"       config overrides: {info['config']}")
        print(f"       use_cache: {info['use_cache']}")


if __name__ == '__main__':
    list_versions()
