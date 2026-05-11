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


@dataclass(frozen=True)
class _SeedState:
    es_key: Any
    train_key: Any
    eval_key: Any


@dataclass(frozen=True)
class _NoiserBundle:
    noiser: Any
    frozen_params: Any
    params: Any


__all__ = ["_EggRollStack", "_NoiserBundle", "_SeedState"]
