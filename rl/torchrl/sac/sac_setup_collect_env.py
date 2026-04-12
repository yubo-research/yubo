from __future__ import annotations

import torchrl.envs as tr_envs
import torchrl.envs.transforms as tr_transforms

from rl.torchrl.dm_control_collect import make_dm_control_collect_env

from .sac_setup_types import _EnvSetup


def _is_dm_control_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("dm_control/")


def _make_collect_env_dm_control_sac(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    seed = int(env_setup.problem_seed) + env_index
    from_pixels = getattr(env_conf, "from_pixels", False)
    pixels_only = getattr(env_conf, "pixels_only", True)
    return make_dm_control_collect_env(
        env_name=str(env_conf.env_name),
        seed=int(seed),
        from_pixels=bool(from_pixels),
        pixels_only=bool(pixels_only),
        tr_envs_module=tr_envs,
        tr_transforms_module=tr_transforms,
        pixels_transform_builder=lambda m: m.Compose(
            m.ToTensorImage(in_keys=["pixels"], out_keys=["observation"], from_int=True),
            m.Resize(84, 84, in_keys=["observation"]),
            m.DoubleToFloat(),
        ),
    )


def _make_collect_env_sac(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    if _is_dm_control_env(env_conf):
        return _make_collect_env_dm_control_sac(env_conf, env_setup, env_index)
    base = env_conf.make()
    seed = int(env_setup.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), tr_transforms.DoubleToFloat())
