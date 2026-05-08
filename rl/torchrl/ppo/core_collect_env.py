from __future__ import annotations

import torchrl.envs as tr_envs
import torchrl.envs.transforms as tr_transforms

from rl.core.pixel_transform import AtariObservationTransform, PixelsToObservation

from .core_utils import _is_atari_env, _is_dm_control_env


def _make_collect_env_atari(env_conf, env_index: int = 0):
    base = env_conf.make()
    seed = int(env_conf.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    transforms = tr_transforms.Compose(AtariObservationTransform(size=84), tr_transforms.DoubleToFloat())
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env_dm_control(env_conf, env_index: int = 0):
    from problems.shimmy_dm_control import make as make_dm_env

    seed = int(env_conf.problem_seed) + env_index
    from_pixels = bool(getattr(env_conf, "from_pixels", False))
    pixels_only = bool(getattr(env_conf, "pixels_only", True))
    base = make_dm_env(env_conf.env_name, from_pixels=from_pixels, pixels_only=pixels_only)
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    if from_pixels:
        transforms = tr_transforms.Compose(PixelsToObservation(size=84), tr_transforms.DoubleToFloat())
    else:
        transforms = tr_transforms.DoubleToFloat()
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env(env_conf, env_index: int = 0):
    if _is_dm_control_env(env_conf):
        return _make_collect_env_dm_control(env_conf, env_index)
    if _is_atari_env(env_conf):
        return _make_collect_env_atari(env_conf, env_index)
    base = env_conf.make()
    seed = int(env_conf.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), tr_transforms.DoubleToFloat())


def _make_collect_env_factory(env_conf, num_envs: int):
    env_index = [0]

    def fn():
        idx = env_index[0]
        env_index[0] = (idx + 1) % num_envs
        return _make_collect_env(env_conf, env_index=idx)

    return fn
