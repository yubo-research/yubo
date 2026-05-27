from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class UHDVectorObjective(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def x0(self) -> np.ndarray: ...

    @property
    def steps_per_episode(self) -> int: ...

    @property
    def num_envs(self) -> int: ...

    def make_policy(self, x: np.ndarray): ...

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]: ...

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]: ...

    def configure_embedding(self, num_probes: int) -> None: ...

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray: ...

    def embed(self, x: np.ndarray) -> np.ndarray: ...

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray: ...


class UHDVectorObjectiveMixin:
    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        from problems.pre_obj_vector_helpers import evaluate_many_serial

        return evaluate_many_serial(self.evaluate, x_batch, seed=seed)

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        from problems.pre_obj_vector_helpers import sample_vector_noise

        return sample_vector_noise(
            dim=self.dim,
            seed=int(seed),
            num_dim_target=num_dim_target,
            num_module_target=num_module_target,
        )


@dataclass(frozen=True)
class BuiltUHDVectorObjective:
    objective: UHDVectorObjective
    source: str
