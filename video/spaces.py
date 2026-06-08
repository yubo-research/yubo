from __future__ import annotations

from typing import Any

import numpy as np


def scale_action_to_space(action: np.ndarray | int, action_space: Any) -> np.ndarray | int:
    """Map policy output to action space. Discrete: pass through as int. Box: scale from [-1,1]."""
    if not hasattr(action_space, "low"):
        if hasattr(action_space, "n"):  # Discrete
            return int(action) if isinstance(action, (int, float, np.integer)) else int(np.asarray(action).item())
        return action
    action = np.asarray(action, dtype=np.float64)
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return np.clip(action, -1.0, 1.0).astype(np.float32)
    return low + (high - low) * (1 + action) / 2


def resolve_max_episode_steps(env_conf: Any) -> int:
    if getattr(env_conf, "gym_conf", None) is not None:
        return int(env_conf.gym_conf.max_steps)
    return int(getattr(env_conf, "max_steps", 99999))
