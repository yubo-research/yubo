from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


@dataclass
class EggRollState:
    """Mutable state for EggRollDesigner."""

    epoch: int = 0
    best_datum: Any = None

    # Normal JAX state
    params: Any = None
    noiser_params: Any = None
    train_key: Any = None
    eval_key: Any = None

    # NanoEgg state
    x: Any = None
    opt_state: Any = None


__all__ = ["_NoiserBundle", "_SeedState", "EggRollState"]
