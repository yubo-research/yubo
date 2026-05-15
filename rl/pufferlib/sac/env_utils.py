from __future__ import annotations

from typing import Any

from rl.core.ppo_envs import is_atari_env_tag, resolve_gym_env_name, to_puffer_game_name
from rl.pufferlib.offpolicy import env_utils

from ...pufferlib_compat import import_pufferlib_modules
from .. import vector_env

ObservationSpec = env_utils.ObservationSpec
EnvSetup = env_utils.EnvSetup
seed_everything = env_utils.seed_everything
resolve_device = env_utils.resolve_device
to_env_action = env_utils.to_env_action
infer_observation_spec = env_utils.infer_observation_spec
prepare_obs_np = env_utils.prepare_obs_np
resolve_backbone_name = env_utils.resolve_backbone_name


build_env_setup = env_utils.build_env_setup


def _make_vector_env_shared(config, **kwargs):
    return vector_env.make_vector_env(config, **kwargs)


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
