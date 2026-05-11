from __future__ import annotations

from collections.abc import Callable

import numpy as np


def evaluate_many_serial(evaluate: Callable, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(x_batch, dtype=np.float64)
    pairs = [evaluate(x, seed=int(seed) + i) for i, x in enumerate(rows)]
    if not pairs:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    mus, ses = zip(*pairs, strict=False)
    return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)


def configure_embedding_indices(dim: int, num_probes: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    n = min(max(int(num_probes), 1), int(dim))
    return rng.choice(int(dim), size=n, replace=False)


def embed_many_with_indices(x_batch: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(x_batch, dtype=np.float64)[:, indices]


def sample_vector_noise(*, dim: int, seed: int, num_dim_target: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    dim = int(dim)
    if num_dim_target is None:
        return rng.standard_normal(dim).astype(np.float64)
    target = float(num_dim_target)
    if target <= 0:
        raise ValueError("perturb target must be > 0.")
    noise = rng.standard_normal(dim).astype(np.float64)
    mask = np.zeros(dim, dtype=bool)
    if 0 < target < 1:
        mask = rng.random(dim) < target
        if not np.any(mask):
            mask[int(rng.integers(dim))] = True
    else:
        k = min(max(int(target), 1), dim)
        mask[rng.choice(dim, size=k, replace=False)] = True
    noise[~mask] = 0.0
    return noise
