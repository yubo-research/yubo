from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class EnvPreprocessingSpec:
    normalize_observation: bool = False
    normalize_reward: bool = False
    reward_normalize_gamma: float = 0.99
    observation_clip: tuple[float, float] | None = None
    reward_clip: tuple[float, float] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.normalize_observation or self.normalize_reward or self.observation_clip is not None or self.reward_clip is not None)


def _as_float_pair(value: Any, *, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        radius = float(value)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{field} scalar clip must be finite and > 0.")
        return -abs(radius), abs(radius)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"{field} clip bounds must be finite.")
        if high <= low:
            raise ValueError(f"{field} clip bounds must satisfy high > low.")
        return low, high
    raise TypeError(f"{field} must be None, scalar radius, or [low, high].")


def _clip_observation_value(value: Any, *, low: float, high: float) -> Any:
    if isinstance(value, dict):
        return {k: _clip_observation_value(v, low=low, high=high) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clip_observation_value(v, low=low, high=high) for v in value)
    if isinstance(value, list):
        return [_clip_observation_value(v, low=low, high=high) for v in value]
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        return value
    clipped = np.clip(arr, low, high)
    if np.isscalar(value):
        return float(clipped)
    return clipped


class _ClipObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, low: float, high: float):
        super().__init__(env)
        self._low = float(low)
        self._high = float(high)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = _clip_observation_value(obs, low=self._low, high=self._high)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = _clip_observation_value(obs, low=self._low, high=self._high)
        return obs, reward, terminated, truncated, info


class _ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, *, low: float, high: float):
        super().__init__(env)
        self._low = float(low)
        self._high = float(high)

    def reward(self, reward):
        return float(np.clip(float(reward), self._low, self._high))


def spec_from_config(config: Any) -> EnvPreprocessingSpec:
    obs_clip_raw = getattr(config, "observation_clip", None)
    if obs_clip_raw is None:
        obs_clip_raw = getattr(config, "obs_clip", None)

    reward_clip_raw = getattr(config, "reward_clip", None)

    spec = EnvPreprocessingSpec(
        normalize_observation=bool(getattr(config, "normalize_observation", False)),
        normalize_reward=bool(getattr(config, "normalize_reward", False)),
        reward_normalize_gamma=float(getattr(config, "reward_normalize_gamma", 0.99)),
        observation_clip=_as_float_pair(obs_clip_raw, field="observation_clip"),
        reward_clip=_as_float_pair(reward_clip_raw, field="reward_clip"),
    )
    gamma = float(spec.reward_normalize_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("reward_normalize_gamma must be a finite positive number.")
    return spec


def spec_from_env_conf(env_conf: Any) -> EnvPreprocessingSpec:
    spec = getattr(env_conf, "_env_preprocessing_spec", None)
    if isinstance(spec, EnvPreprocessingSpec):
        return spec
    return EnvPreprocessingSpec()


def attach_env_preprocessing(env_conf: Any, source: Any) -> Any:
    spec = source if isinstance(source, EnvPreprocessingSpec) else spec_from_config(source)
    setattr(env_conf, "_env_preprocessing_spec", spec)
    return env_conf


def apply_gym_preprocessing(
    env: gym.Env,
    *,
    env_conf: Any | None = None,
    preprocess_spec: EnvPreprocessingSpec | None = None,
) -> gym.Env:
    spec = preprocess_spec if preprocess_spec is not None else spec_from_env_conf(env_conf)
    if not spec.enabled:
        return env

    wrapped = env
    if bool(spec.normalize_observation):
        wrapped = gym.wrappers.NormalizeObservation(wrapped)
    if spec.observation_clip is not None:
        low, high = spec.observation_clip
        wrapped = _ClipObservationWrapper(wrapped, low=low, high=high)

    if bool(spec.normalize_reward):
        wrapped = gym.wrappers.NormalizeReward(wrapped, gamma=float(spec.reward_normalize_gamma))
    if spec.reward_clip is not None:
        low, high = spec.reward_clip
        wrapped = _ClipRewardWrapper(wrapped, low=low, high=high)
    return wrapped
