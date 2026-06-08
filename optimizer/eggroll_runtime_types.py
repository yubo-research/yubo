from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _EggRollStack:
    jax: Any
    jnp: Any
    optax: Any
    simple_es_tree_key: Any
    all_noisers: dict[str, Any]


__all__ = ["_EggRollStack"]
