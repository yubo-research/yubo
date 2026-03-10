from __future__ import annotations

from typing import Any, Callable


def make_dm_control_collect_env(
    *,
    env_name: str,
    seed: int,
    from_pixels: bool,
    pixels_only: bool,
    tr_envs_module: Any,
    tr_transforms_module: Any,
    pixels_transform_builder: Callable[[Any], Any],
):
    from dm_control import suite

    from problems.dm_control_env import _configure_headless_render_backend, _parse_env_name

    _configure_headless_render_backend("rgb_array" if bool(from_pixels) else None)
    domain, task = _parse_env_name(str(env_name))
    dm_env = suite.load(domain, task, task_kwargs={"random": int(seed)})
    base = tr_envs_module.DMControlWrapper(dm_env, from_pixels=bool(from_pixels), pixels_only=bool(pixels_only))

    if bool(from_pixels):
        transforms = pixels_transform_builder(tr_transforms_module)
    else:
        obs_keys = list(base.observation_spec.keys(True, True))
        cat = tr_transforms_module.CatTensors(in_keys=obs_keys, out_key="observation", del_keys=False, sort=False)
        transforms = tr_transforms_module.Compose(cat, tr_transforms_module.DoubleToFloat())
    return tr_envs_module.TransformedEnv(base, transforms)
