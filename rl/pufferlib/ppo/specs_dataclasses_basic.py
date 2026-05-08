from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class _ObservationSpec:
    mode: str
    raw_shape: tuple[int, ...]
    vector_dim: int | None = None
    channels: int | None = None
    image_size: int | None = None


@dataclasses.dataclass(frozen=True)
class _ActionSpec:
    kind: str
    dim: int
    low: np.ndarray | None = None
    high: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class _TrainPlan:
    num_envs: int
    num_steps: int
    batch_size: int
    minibatch_size: int
    num_iterations: int
