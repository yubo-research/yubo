"""Shared observation normalizer for vector envs (main process)."""

from __future__ import annotations

import numpy as np


class SharedObsNormalizer:
    """Running mean/std normalizer. Updates from batches, normalizes with shared stats."""

    def __init__(self, obs_dim: int, epsilon: float = 1e-8):
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._var = np.ones(obs_dim, dtype=np.float64)
        self._count = epsilon

    def update(self, obs: np.ndarray) -> None:
        """Update running stats with batch of observations."""
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        batch = obs.reshape(-1, self._mean.size)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._mean = self._mean + delta * batch_count / total
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self._count * batch_count / total
        self._var = m2 / total
        self._count = total

    def normalize(self, obs: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize observations. Does not update stats. Clips to [-clip, clip] (CleanRL)."""
        obs = np.asarray(obs, dtype=np.float32)
        std = np.sqrt(self._var + 1e-8).astype(np.float32)
        out = ((obs - self._mean.astype(np.float32)) / std).astype(np.float32)
        return np.clip(out, -clip, clip)
