from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from problems.pre_obj_specs import resolve_nanoegg_pretrain_spec
from problems.pre_obj_vector_helpers import (
    configure_embedding_indices,
    embed_many_with_indices,
    evaluate_many_serial,
    sample_vector_noise,
)


class NanoEggPretrainVectorObjective:
    """Yubo-owned JAX NanoEgg objective adapter for UHD optimizers."""

    def __init__(self, cfg: Any) -> None:
        from problems.nanoegg_obj import build_nanoegg_uhd_objective

        self.cfg = cfg
        self.spec = resolve_nanoegg_pretrain_spec(cfg.env_tag, cfg.policy_tag)
        self._objective = build_nanoegg_uhd_objective(cfg=cfg, spec=self.spec)
        self._vectorize = bool(getattr(self._objective, "_vectorize", False))
        self._embed_indices: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(getattr(self._objective, "dim"))

    @property
    def x0(self) -> np.ndarray:
        return np.asarray(getattr(self._objective, "x0"), dtype=np.float64)

    @property
    def steps_per_episode(self) -> int:
        return int(getattr(self._objective, "steps_per_episode", 1))

    @property
    def num_envs(self) -> int:
        return int(getattr(self._objective, "num_envs", 1))

    def make_policy(self, x: np.ndarray):
        make_policy = getattr(self._objective, "make_policy", None)
        if callable(make_policy):
            return make_policy(x)
        return SimpleNamespace(_nanoegg_pretrain_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        result = self._objective.evaluate(np.asarray(x, dtype=np.float64), seed=int(seed))
        if isinstance(result, tuple) and len(result) == 2:
            return float(result[0]), float(result[1])
        return float(result), 0.0

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        evaluate_many = getattr(self._objective, "evaluate_many", None)
        if callable(evaluate_many):
            mus, ses = evaluate_many(np.asarray(x_batch, dtype=np.float64), seed=int(seed))
            return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)
        return evaluate_many_serial(self.evaluate, x_batch, seed=seed)

    def evaluate_many_common_seed(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        evaluate_many = getattr(self._objective, "evaluate_many_common_seed", None)
        if callable(evaluate_many):
            mus, ses = evaluate_many(np.asarray(x_batch, dtype=np.float64), seed=int(seed))
            return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)
        return self.evaluate_many(x_batch, seed=seed)

    def configure_embedding(self, num_probes: int) -> None:
        configure_embedding = getattr(self._objective, "configure_embedding", None)
        if callable(configure_embedding):
            configure_embedding(int(num_probes))
            return
        self._embed_indices = configure_embedding_indices(self.dim, num_probes)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        embed_many = getattr(self._objective, "embed_many", None)
        if callable(embed_many):
            return np.asarray(embed_many(np.asarray(x_batch, dtype=np.float64)), dtype=np.float64)
        if self._embed_indices is None:
            self.configure_embedding(64)
        assert self._embed_indices is not None
        return embed_many_with_indices(x_batch, self._embed_indices)

    def embed(self, x: np.ndarray) -> np.ndarray:
        embed = getattr(self._objective, "embed", None)
        if callable(embed):
            return np.asarray(embed(np.asarray(x, dtype=np.float64)), dtype=np.float64)
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        sample_noise = getattr(self._objective, "sample_noise", None)
        if callable(sample_noise):
            return np.asarray(
                sample_noise(
                    seed=int(seed),
                    num_dim_target=num_dim_target,
                    num_module_target=num_module_target,
                ),
                dtype=np.float64,
            )
        target = num_module_target if num_module_target is not None else num_dim_target
        return sample_vector_noise(dim=self.dim, seed=int(seed), num_dim_target=target)

    def sample_eggroll_noiser_noise(self, x: np.ndarray, **kwargs) -> np.ndarray:
        sample = getattr(self._objective, "sample_eggroll_noiser_noise", None)
        if not callable(sample):
            raise ValueError("NanoEgg objective does not expose EggRoll noiser perturbation materialization.")
        return np.asarray(sample(np.asarray(x, dtype=np.float64), **kwargs), dtype=np.float64)

    def close(self) -> None:
        close = getattr(self._objective, "close", None)
        if callable(close):
            close()
