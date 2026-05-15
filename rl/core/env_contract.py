from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from common.env_tags import is_atari_env_tag
from rl.core.continuous_actions import normalize_action_bounds

ObsMode = Literal["vector", "pixels"]
ActionKind = Literal["discrete", "continuous"]


@dataclass(frozen=True)
class ObservationContract:
    mode: ObsMode
    raw_shape: tuple[int, ...]
    vector_dim: int | None = None
    model_channels: int | None = None
    image_size: int | None = None


@dataclass(frozen=True)
class ActionContract:
    kind: ActionKind
    dim: int
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class EnvIOContract:
    observation: ObservationContract
    action: ActionContract


def _space_shape(space: Any) -> tuple[int, ...]:
    shape = getattr(space, "shape", None)
    if shape is None:
        return ()
    return tuple((int(v) for v in shape))


def resolve_observation_contract(env_conf: Any, *, default_image_size: int = 84) -> ObservationContract:
    state_space = getattr(env_conf, "state_space", None)
    if state_space is None:
        gym_conf = getattr(env_conf, "gym_conf", None)
        state_space = getattr(gym_conf, "state_space", None)
    if state_space is None:
        raise ValueError("Observation space is missing on env_conf. Call env_conf.ensure_spaces() before resolving env contracts.")

    raw_shape = _space_shape(state_space)
    from_pixels = bool(getattr(env_conf, "from_pixels", False))

    if not from_pixels:
        vector_dim = int(np.prod(raw_shape)) if raw_shape else 1
        return ObservationContract(mode="vector", raw_shape=raw_shape, vector_dim=vector_dim)

    # Pixel mode: avoid fragile heuristics. Assume 84x84 and 3 channels unless tagged/specified.
    image_size = int(getattr(env_conf, "image_size", default_image_size))

    # Most RL CNNs (Nature CNN) expect 3 channels (RGB) or 4 channels (Atari FrameStack).
    # We check the tag if available on env_conf, but also look at the shape directly.
    env_tag = str(getattr(env_conf, "env_tag", ""))
    model_channels = 4 if is_atari_env_tag(env_tag) else 3

    # If the shape clearly indicates 4 channels, use it (handles FrameStack without Atari tag)
    if len(raw_shape) >= 3:
        if int(raw_shape[0]) == 4 or int(raw_shape[-1]) == 4:
            model_channels = 4

    # Allow explicit override on env_conf
    model_channels = int(getattr(env_conf, "model_channels", model_channels))

    return ObservationContract(
        mode="pixels",
        raw_shape=raw_shape,
        model_channels=model_channels,
        image_size=image_size,
    )


def resolve_action_contract(action_space: Any) -> ActionContract:
    is_discrete = hasattr(action_space, "n") and (not hasattr(action_space, "shape") or len(getattr(action_space, "shape", ())) == 0)
    if is_discrete:
        dim = int(action_space.n)
        return ActionContract(
            kind="discrete",
            dim=dim,
            low=np.array([0.0], dtype=np.float32),
            high=np.array([float(dim - 1)], dtype=np.float32),
        )
    shape = _space_shape(action_space)
    dim = int(np.prod(shape)) if shape else 1
    low, high = normalize_action_bounds(action_space.low, action_space.high, dim)
    return ActionContract(kind="continuous", dim=dim, low=low, high=high)


def resolve_env_io_contract(env_conf: Any, *, default_image_size: int = 84) -> EnvIOContract:
    return EnvIOContract(
        observation=resolve_observation_contract(env_conf, default_image_size=int(default_image_size)),
        action=resolve_action_contract(env_conf.action_space),
    )


def resolve_backbone_name(default_backbone_name: str, observation: ObservationContract) -> str:
    if observation.mode == "pixels":
        if int(observation.model_channels or 3) == 4:
            return "nature_cnn_atari"
        return "nature_cnn"
    return str(default_backbone_name)
