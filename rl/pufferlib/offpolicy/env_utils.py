from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from common.env_tags import is_atari_env_tag, normalize_dm_control_tag
from problems.problem import build_problem
from rl.core import env_setup, runtime
from rl.core.continuous_actions import scale_action_to_env
from rl.core.ppo_envs import (
    _maybe_register_atari_dm_backends,
    resolve_gym_env_name,
    to_puffer_game_name,
)

from ...pufferlib_compat import import_pufferlib_modules
from .. import vector_env
from . import backbone_name
from .pixel_utils import ensure_pixel_obs_format
from .runtime_utils import obs_scale_from_env, select_device


@dataclasses.dataclass(frozen=True)
class ObservationSpec:
    mode: str
    raw_shape: tuple[int, ...]
    vector_dim: int | None = None
    channels: int | None = None
    image_size: int | None = None


@dataclasses.dataclass(frozen=True)
class EnvSetup:
    env_conf: Any
    problem_seed: int
    noise_seed_0: int
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


def seed_everything(seed: int) -> None:
    runtime.seed_everything(int(seed))


def resolve_device(device: str) -> torch.device:
    return select_device(str(device))


def to_env_action(action_norm: np.ndarray, *, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return scale_action_to_env(action_norm, low, high, clip=True)


# TODO: UNIFY HEURISTICS. The functions _infer_channels and _infer_image_size below are duplicates of logic
# that was recently cleaned up in rl.core.env_contract. During the next sweep, these should be removed
# in favor of resolve_observation_contract() and the shared tag utils in common.env_tags.
def _infer_channels(shape: tuple[int, ...], *, fallback: int) -> int:
    if len(shape) >= 3:
        if int(shape[0]) in (1, 3, 4):
            return int(shape[0])
        if int(shape[-1]) in (1, 3, 4):
            return int(shape[-1])
    if len(shape) == 2 and int(fallback) > 1:
        return int(fallback)
    return 3


def _infer_image_size(shape: tuple[int, ...], *, default_size: int) -> int:
    if len(shape) >= 3:
        if int(shape[0]) in (1, 3, 4):
            return int(min(shape[1], shape[2]))
        if int(shape[-1]) in (1, 3, 4):
            return int(min(shape[0], shape[1]))
        return int(min(shape[-2], shape[-1]))
    if len(shape) == 2:
        return int(min(shape[0], shape[1]))
    return int(default_size)


def infer_observation_spec(config: Any, obs_np: np.ndarray) -> ObservationSpec:
    obs_arr = np.asarray(obs_np)
    if obs_arr.ndim == 0:
        raise ValueError("Observation must include at least one dimension.")
    raw_shape = tuple((int(v) for v in (obs_arr.shape[1:] if obs_arr.ndim >= 2 else obs_arr.shape)))
    backbone_key = str(config.backbone_name).strip().lower()
    looks_like_pixels = bool(config.from_pixels) or obs_arr.ndim >= 4 or "nature_cnn" in backbone_key
    if looks_like_pixels:
        channels = _infer_channels(raw_shape, fallback=max(1, int(config.framestack)))
        image_size = _infer_image_size(raw_shape, default_size=84)
        return ObservationSpec(mode="pixels", raw_shape=raw_shape, channels=channels, image_size=image_size)
    vector_dim = int(np.prod(raw_shape)) if raw_shape else 1
    return ObservationSpec(mode="vector", raw_shape=raw_shape, vector_dim=vector_dim)


def prepare_obs_np(obs_np: np.ndarray, *, obs_spec: ObservationSpec) -> np.ndarray:
    obs_arr = np.asarray(obs_np)
    if obs_spec.mode == "pixels":
        obs_t = ensure_pixel_obs_format(
            torch.as_tensor(obs_arr),
            channels=int(obs_spec.channels or 3),
            size=int(obs_spec.image_size or 84),
            scale_float_255=True,
        )
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        return np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
    vec = np.asarray(obs_arr, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    elif vec.ndim > 2:
        vec = vec.reshape(vec.shape[0], -1)
    return vec


def continuous_gym_runtime_from_problem(
    env_tag: str,
    *,
    problem_seed: int,
    noise_seed_0: int,
    from_pixels: bool,
    pixels_only: bool,
):
    _maybe_register_atari_dm_backends(env_tag)
    adj = normalize_dm_control_tag(env_tag, from_pixels=from_pixels)
    # policy_tag="linear" is a placeholder; only problem.env is used (policy is never built)
    problem = build_problem(adj, "linear", problem_seed=int(problem_seed), noise_seed_0=int(noise_seed_0))
    env = problem.env
    if env.spec.env_name.startswith("dm_control/"):
        env.spec.pixels_only = bool(pixels_only)
    return env


def resolve_backbone_name(config: Any, obs_spec: ObservationSpec) -> str:
    return backbone_name.resolve_backbone_name(config, obs_spec)


def build_env_setup(config: Any) -> EnvSetup:
    shared = env_setup.build_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(config.from_pixels),
        pixels_only=bool(config.pixels_only),
        get_env_conf_fn=continuous_gym_runtime_from_problem,
        include_continuous_info=True,
        obs_scale_from_env_fn=obs_scale_from_env,
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
    return vector_env.make_vector_env(config, **kwargs)


def make_vector_env(config: Any):
    return _make_vector_env_shared(
        config,
        import_pufferlib_modules_fn=import_pufferlib_modules,
        is_atari_env_tag_fn=is_atari_env_tag,
        to_puffer_game_name_fn=to_puffer_game_name,
        resolve_gym_env_name_fn=resolve_gym_env_name,
    )
