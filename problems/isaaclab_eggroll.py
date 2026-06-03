from __future__ import annotations

import math
from typing import Any

import numpy as np

from optimizer.eggroll_runtime_noise import sample_leaf_mask


class EggRollIsaacCodec:
    def __init__(self, policy: Any) -> None:
        if not all(hasattr(policy, name) for name in ("get_params", "set_params", "clone", "num_params")):
            raise TypeError("IsaacLab scoring requires a flat-parameter policy with get_params, set_params, clone, and num_params.")
        self.dim = int(policy.num_params())
        offsets: list[int] = []
        sizes: list[int] = []
        shapes: list[tuple[int, ...]] = []
        start = 0
        params = list(policy.parameters()) if hasattr(policy, "parameters") else []
        if params:
            for param in params:
                size = int(param.numel())
                offsets.append(start)
                sizes.append(size)
                shapes.append(tuple(int(v) for v in param.shape))
                start += size
        else:
            offsets.append(0)
            sizes.append(self.dim)
            shapes.append((self.dim,))
        self.offsets = tuple(offsets)
        self.sizes = tuple(sizes)
        self.shapes = tuple(shapes)

    def initial(self, policy: Any) -> np.ndarray:
        x = np.asarray(policy.get_params(), dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Policy returned {x.size} params, expected {self.dim}.")
        return x

    def load(self, policy: Any, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Candidate has {x.size} params, expected {self.dim}.")
        policy.set_params(np.clip(x, -1.0, 1.0))


class EggRollIsaacNoise:
    def __init__(self, codec: EggRollIsaacCodec) -> None:
        self._codec = codec

    def sample(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        if num_module_target is not None:
            return self._sample_leaf_noise(rng, float(num_module_target))
        if num_dim_target is not None:
            return self._sample_dim_noise(rng, float(num_dim_target))
        return rng.standard_normal(int(self._codec.dim)).astype(np.float64)

    def _sample_dim_noise(self, rng: np.random.Generator, target: float) -> np.ndarray:
        if target <= 0:
            raise ValueError("dim perturb target must be > 0.")
        dim = int(self._codec.dim)
        noise = np.zeros(dim, dtype=np.float64)
        if 0 < target < 1:
            mask = rng.random(dim) < target
            if not np.any(mask):
                mask[int(rng.integers(dim))] = True
            noise[mask] = rng.standard_normal(int(np.count_nonzero(mask)))
            return noise
        k = min(max(int(target), 1), dim)
        idx = rng.choice(dim, size=k, replace=False)
        noise[idx] = rng.standard_normal(k)
        return noise

    def _sample_leaf_noise(self, rng: np.random.Generator, target: float) -> np.ndarray:
        if target <= 0:
            raise ValueError("module perturb target must be > 0.")
        mask = sample_leaf_mask(rng, num_leaves=len(self._codec.sizes), target=float(target))
        noise = np.zeros(int(self._codec.dim), dtype=np.float64)
        for selected, start, size in zip(mask, self._codec.offsets, self._codec.sizes, strict=True):
            if selected:
                end = int(start) + int(size)
                noise[int(start) : end] = rng.standard_normal(int(size))
        return noise


def sample_rank_noise(
    codec: EggRollIsaacCodec,
    *,
    seed: int,
    rank: int,
    group_size: int,
    freeze_nonlora: bool,
) -> np.ndarray:
    if int(rank) < 1:
        raise ValueError("IsaacLab EggRoll rank must be >= 1.")
    if int(group_size) < 0:
        raise ValueError("IsaacLab EggRoll group_size must be >= 0.")
    noise = np.zeros(int(codec.dim), dtype=np.float64)
    leaf_seeds = np.random.SeedSequence(int(seed)).spawn(len(codec.sizes))
    leaves = zip(codec.offsets, codec.sizes, codec.shapes, leaf_seeds, strict=True)
    for start, size, shape, leaf_seed in leaves:
        end = int(start) + int(size)
        noise[int(start) : end] = _sample_leaf_rank_noise(
            np.random.default_rng(leaf_seed),
            shape=tuple(shape),
            rank=int(rank),
            freeze_nonlora=bool(freeze_nonlora),
        ).reshape(-1)
    return noise


def _sample_leaf_rank_noise(
    rng: np.random.Generator,
    *,
    shape: tuple[int, ...],
    rank: int,
    freeze_nonlora: bool,
) -> np.ndarray:
    if len(shape) < 2:
        if freeze_nonlora:
            return np.zeros(shape, dtype=np.float64)
        return rng.standard_normal(shape).astype(np.float64)
    rows = int(shape[0])
    cols = int(np.prod(shape[1:], dtype=np.int64))
    left = rng.standard_normal((rows, int(rank)))
    right = rng.standard_normal((cols, int(rank)))
    return (left @ right.T / math.sqrt(float(rank))).reshape(shape).astype(np.float64)
