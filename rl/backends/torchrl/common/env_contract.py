from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

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
    return tuple(int(v) for v in shape)


def _infer_raw_channels(shape: tuple[int, ...]) -> int | None:
    if not shape:
        return None
    dims = [int(v) for v in shape]
    if 4 in dims:
        return 4
    if 3 in dims:
        return 3
    if 1 in dims:
        return 1
    return None


def _infer_image_size(shape: tuple[int, ...], default_size: int) -> int:
    if len(shape) >= 2:
        for i in range(len(shape) - 1):
            if int(shape[i]) == int(shape[i + 1]) and int(shape[i]) >= 16:
                return int(shape[i])
    large_dims = [int(v) for v in shape if int(v) >= 16]
    if len(large_dims) >= 2:
        return int(min(large_dims[-2], large_dims[-1]))
    if len(large_dims) == 1:
        return int(large_dims[0])
    return int(default_size)


def resolve_observation_contract(env_conf: Any, *, default_image_size: int = 84) -> ObservationContract:
    gym_conf = getattr(env_conf, "gym_conf", None)
    state_space = getattr(gym_conf, "state_space", None) if gym_conf is not None else None
    raw_shape = _space_shape(state_space)
    from_pixels = bool(getattr(env_conf, "from_pixels", False))

    if not from_pixels:
        vector_dim = int(np.prod(raw_shape)) if raw_shape else 1
        return ObservationContract(
            mode="vector",
            raw_shape=raw_shape,
            vector_dim=vector_dim,
        )

    raw_channels = _infer_raw_channels(raw_shape)
    model_channels = 4 if raw_channels == 4 else 3
    image_size = _infer_image_size(raw_shape, int(default_image_size))
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
    return ActionContract(
        kind="continuous",
        dim=dim,
        low=np.asarray(action_space.low, dtype=np.float32),
        high=np.asarray(action_space.high, dtype=np.float32),
    )


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
