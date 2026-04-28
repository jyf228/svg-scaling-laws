"""
Central config loader.
"""

import functools
from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).parents[2] / "configs"


@functools.lru_cache(maxsize=None)
def get_config(name: str) -> dict[str, Any]:
    """Load and cache a YAML config file by name."""
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{name}' not found. Expected: {path}\n"
            f"Available configs: {[p.stem for p in _CONFIGS_DIR.glob('*.yaml')]}"
        )
    with open(path) as f:
        return yaml.safe_load(f)
