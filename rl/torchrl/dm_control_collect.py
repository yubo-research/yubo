from __future__ import annotations

from typing import Any, Callable

from common.obs_mode import obs_mode_pixels_only, obs_mode_uses_pixels


def make_dm_control_collect_env(
    *,
    env_name: str,
    seed: int,
    obs_mode: str,
    tr_envs_module: Any,
    tr_transforms_module: Any,
    pixels_transform_builder: Callable[[Any], Any],
):
    from problems.dm_control_env import load

    uses_pixels = obs_mode_uses_pixels(str(obs_mode))
    dm_env = load(str(env_name), seed=int(seed), render_mode="rgb_array" if uses_pixels else None)
    base = tr_envs_module.DMControlWrapper(dm_env, from_pixels=uses_pixels, pixels_only=obs_mode_pixels_only(str(obs_mode)))

    if uses_pixels:
        transforms = pixels_transform_builder(tr_transforms_module)
    else:
        obs_keys = list(base.observation_spec.keys(True, True))
        cat = tr_transforms_module.CatTensors(in_keys=obs_keys, out_key="observation", del_keys=False, sort=False)
        transforms = tr_transforms_module.Compose(cat, tr_transforms_module.DoubleToFloat())
    return tr_envs_module.TransformedEnv(base, transforms)
