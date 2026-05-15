from __future__ import annotations

from typing import Any

import torchrl.envs as tr_envs
import torchrl.envs.transforms as tr_transforms


def _gym_wrapper_without_isaaclab_probe(base):
    import torchrl.envs.libs.gym as torchrl_gym

    if hasattr(torchrl_gym, "_has_isaaclab"):
        torchrl_gym._has_isaaclab = False
    return tr_envs.GymWrapper(base)


def make_collect_env(env_conf: Any, *, env_index: int = 0):
    """Unified creation of a TorchRL-compatible collection environment."""
    # 1. Use the core unified Gym creator (handles pixels, skip, clip, normalization)
    seed = int(getattr(env_conf, "problem_seed", 0)) + env_index
    base = env_conf.make_gym_env(seed=seed)

    # 2. TorchRL wrapping
    wrapped = _gym_wrapper_without_isaaclab_probe(base)

    # 3. Standard transforms (Always Float32)
    return tr_envs.TransformedEnv(wrapped, tr_transforms.DoubleToFloat())
