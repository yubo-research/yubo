from __future__ import annotations

import numpy as np

from rl.core.continuous_actions import scale_action_to_env, unscale_action_from_env


def _scale_action_to_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return scale_action_to_env(action, action_low, action_high, clip=False)


def _unscale_action_from_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return unscale_action_from_env(action, action_low, action_high, clip=True)
