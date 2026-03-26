from __future__ import annotations

from typing import Any

from rl.core.env_setup import build_continuous_gym_env_setup
from rl.core.ppo_envs import is_atari_env_tag, resolve_gym_env_name, to_puffer_game_name
from rl.pufferlib.offpolicy import env_utils as _impl

from ...pufferlib_compat import import_pufferlib_modules
from ..vector_env import make_vector_env as _make_vector_env_common

ObservationSpec = _impl.ObservationSpec
EnvSetup = _impl.EnvSetup
seed_everything = _impl.seed_everything
resolve_device = _impl.resolve_device
to_env_action = _impl.to_env_action
infer_observation_spec = _impl.infer_observation_spec
prepare_obs_np = _impl.prepare_obs_np
resolve_backbone_name = _impl.resolve_backbone_name


def build_env_setup(config: Any) -> EnvSetup:
    shared = build_continuous_gym_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(config.from_pixels),
        pixels_only=bool(config.pixels_only),
        get_env_conf_fn=_impl.continuous_gym_runtime_from_problem,
        obs_scale_from_env_fn=_impl.obs_scale_from_env,
    )
    return EnvSetup(
        env_conf=shared.env_conf,
        problem_seed=int(shared.problem_seed),
        noise_seed_0=int(shared.noise_seed_0),
        obs_lb=shared.obs_lb,
        obs_width=shared.obs_width,
        act_dim=int(shared.act_dim),
        action_low=shared.action_low,
        action_high=shared.action_high,
    )


def _make_vector_env_shared(config, **kwargs):
    return _make_vector_env_common(config, **kwargs)


def make_vector_env(config: Any):
    return _make_vector_env_shared(
        config,
        import_pufferlib_modules_fn=import_pufferlib_modules,
        is_atari_env_tag_fn=is_atari_env_tag,
        to_puffer_game_name_fn=to_puffer_game_name,
        resolve_gym_env_name_fn=resolve_gym_env_name,
    )


__all__ = [
    "EnvSetup",
    "ObservationSpec",
    "build_continuous_gym_env_setup",
    "build_env_setup",
    "infer_observation_spec",
    "is_atari_env_tag",
    "make_vector_env",
    "prepare_obs_np",
    "resolve_backbone_name",
    "resolve_device",
    "resolve_gym_env_name",
    "seed_everything",
    "to_env_action",
    "to_puffer_game_name",
]
