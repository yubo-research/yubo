from __future__ import annotations

from typing import Any

import numpy as np


def _reshape_batch_obs(obs: np.ndarray, host: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    n = int(getattr(host, "num_envs", 1) or 1)
    if n <= 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and int(arr.shape[0]) == n:
        return arr.astype(np.float32, copy=False)
    shape = tuple(int(v) for v in getattr(host.observation_space, "shape", ()))
    if len(shape) == 1:
        flat = int(shape[0])
        if n > 1 and flat % n == 0 and flat != n:
            per_env_dim = flat // n
        else:
            per_env_dim = flat
    elif shape and shape[0] == n:
        per_env_dim = int(np.prod(shape[1:]))
    else:
        per_env_dim = max(1, int(arr.size) // n)
    if arr.ndim == 1 and arr.size == n * per_env_dim:
        return arr.reshape(n, per_env_dim)
    return arr.reshape(n, per_env_dim)


def host_reset_batch(host: Any, seed: int) -> np.ndarray:
    obs, _info = host.reset_batch(seed=int(seed))
    return _reshape_batch_obs(obs, host)


def host_step_batch(
    host: Any,
    actions: np.ndarray,
    episode_steps: np.ndarray,
    *,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs, reward, terminated, truncated, _info = host.step_batch(np.asarray(actions, dtype=np.float32))
    obs = _reshape_batch_obs(obs, host)
    reward = np.asarray(reward, dtype=np.float32).reshape(-1)
    terminated = np.asarray(terminated, dtype=bool).reshape(-1)
    truncated = np.asarray(truncated, dtype=bool).reshape(-1)
    steps = np.asarray(episode_steps, dtype=np.int32).reshape(-1)
    next_steps = np.minimum(steps + 1, int(max_steps))
    terminated = terminated | (next_steps >= int(max_steps))
    return obs, reward, terminated.astype(np.float32), truncated.astype(np.float32), next_steps.astype(np.int32)
