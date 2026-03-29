from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import numpy as np
import torch

from rl.core.continuous_actions import scale_action_to_env
from rl.core.env_setup import build_continuous_gym_env_setup
from rl.core.ppo_envs import is_atari_env_tag, resolve_gym_env_name, to_puffer_game_name
from rl.core.runtime import seed_everything as _seed_everything_core

from ...pufferlib_compat import import_pufferlib_modules
from ..vector_env import make_vector_env as _make_vector_env_common
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
    _seed_everything_core(int(seed))


def resolve_device(device: str) -> torch.device:
    return select_device(str(device))


def to_env_action(action_norm: np.ndarray, *, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return scale_action_to_env(action_norm, low, high, clip=True)


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


def resolve_backbone_name(config: Any, obs_spec: ObservationSpec) -> str:
    if obs_spec.mode != "pixels":
        return str(config.backbone_name)
    channels = int(obs_spec.channels or 3)
    key = str(config.backbone_name).strip().lower()
    if key in {"mlp", "nature_cnn"} and channels == 4:
        return "nature_cnn_atari"
    if key in {"mlp", "nature_cnn_atari"} and channels != 4:
        return "nature_cnn"
    return str(config.backbone_name)


def build_env_setup(config: Any) -> EnvSetup:
    get_env_conf = importlib.import_module("problems.env_conf").get_env_conf
    shared = build_continuous_gym_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(config.from_pixels),
        pixels_only=bool(config.pixels_only),
        get_env_conf_fn=get_env_conf,
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
    return _make_vector_env_common(config, **kwargs)


def make_vector_env(config: Any):
    return _make_vector_env_shared(
        config,
        import_pufferlib_modules_fn=import_pufferlib_modules,
        is_atari_env_tag_fn=is_atari_env_tag,
        to_puffer_game_name_fn=to_puffer_game_name,
        resolve_gym_env_name_fn=resolve_gym_env_name,
    )
