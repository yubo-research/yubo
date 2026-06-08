from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        if "policy" in obs:
            return _flatten_obs(obs["policy"])
        if "observation" in obs:
            return _flatten_obs(obs["observation"])
        parts = [_flatten_obs(obs[key]).ravel() for key in sorted(obs)]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

    arr = _as_numpy(obs).astype(np.float32, copy=False)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    return np.ravel(arr).astype(np.float32, copy=False)


def _flatten_obs_batch(obs: Any, *, num_envs: int) -> np.ndarray:
    if isinstance(obs, dict):
        if "policy" in obs:
            return _flatten_obs_batch(obs["policy"], num_envs=num_envs)
        if "observation" in obs:
            return _flatten_obs_batch(obs["observation"], num_envs=num_envs)
        parts = [_flatten_obs_batch(obs[key], num_envs=num_envs) for key in sorted(obs)]
        return np.concatenate(parts, axis=1) if parts else np.zeros((int(num_envs), 0), dtype=np.float32)

    arr = _as_numpy(obs).astype(np.float32, copy=False)
    if arr.ndim == 0:
        return np.full((int(num_envs), 1), float(arr), dtype=np.float32)
    if arr.ndim == 1:
        if int(num_envs) == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(int(num_envs), -1)
    elif arr.shape[0] != int(num_envs):
        arr = np.reshape(arr, (int(num_envs), -1))
    else:
        arr = arr.reshape(int(num_envs), -1)
    return arr.astype(np.float32, copy=False)


def _flat_box_from_space(space: Any):
    from gymnasium import spaces

    if hasattr(space, "spaces"):
        if "policy" in space.spaces:
            return _flat_box_from_space(space.spaces["policy"])
        if "observation" in space.spaces:
            return _flat_box_from_space(space.spaces["observation"])
        boxes = [_flat_box_from_space(space.spaces[key]) for key in sorted(space.spaces)]
        low = np.concatenate([np.ravel(np.asarray(box.low, dtype=np.float32)) for box in boxes], axis=0)
        high = np.concatenate([np.ravel(np.asarray(box.high, dtype=np.float32)) for box in boxes], axis=0)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.ravel(np.asarray(space.low, dtype=np.float32))
        high = np.ravel(np.asarray(space.high, dtype=np.float32))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    if hasattr(space, "shape"):
        size = int(np.prod(tuple(int(v) for v in space.shape)))
        return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    raise TypeError(f"Unsupported Isaac Lab observation space type: {type(space).__name__}")


def _single_box_space(space: Any, *, num_envs: int):
    if not hasattr(space, "low") or not hasattr(space, "high") or not hasattr(space, "shape"):
        return space
    shape = tuple(int(v) for v in space.shape)
    if len(shape) <= 1 or shape[0] != int(num_envs):
        return space
    from gymnasium import spaces

    low = np.asarray(space.low, dtype=np.float32)[0]
    high = np.asarray(space.high, dtype=np.float32)[0]
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _single_action_space(env: Any, *, num_envs: int):
    space = getattr(env, "single_action_space", None) or getattr(env, "action_space")
    return _single_box_space(space, num_envs=int(num_envs))


def _single_observation_space(env: Any, *, num_envs: int):
    space = getattr(env, "single_observation_space", None) or getattr(env, "observation_space")
    return _single_box_space(space, num_envs=int(num_envs))


def _scalar_bool(value: Any) -> bool:
    arr = _as_numpy(value)
    if arr.size == 0:
        return False
    return bool(np.ravel(arr)[0])


def _scalar_float(value: Any) -> float:
    arr = _as_numpy(value)
    if arr.size == 0:
        return 0.0
    return float(np.ravel(arr)[0])


def _adapter_spaces(env: Any, *, num_envs: int):
    observation_space = _flat_box_from_space(_single_observation_space(env, num_envs=int(num_envs)))
    action_space = _single_action_space(env, num_envs=int(num_envs))
    return observation_space, action_space


class _IsaacLabAdapterBase:
    def _init_adapter(self, env: Any, *, num_envs: int) -> None:
        self._env = env
        self._num_envs = int(num_envs)
        self.observation_space, self.action_space = _adapter_spaces(env, num_envs=self._num_envs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


class IsaacLabGymEnvAdapter(_IsaacLabAdapterBase):
    def __init__(self, env: Any, *, num_envs: int = 1) -> None:
        self._init_adapter(env, num_envs=int(num_envs))

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        return _flatten_obs(obs), info

    def step(self, action):
        step_out = self._env.step(self._format_action(action))
        if len(step_out) != 5:
            raise ValueError(f"Unsupported Isaac Lab step return arity: {len(step_out)}")
        obs, reward, terminated, truncated, info = step_out
        result = (
            _flatten_obs(obs),
            _scalar_float(reward),
            _scalar_bool(terminated),
            _scalar_bool(truncated),
            info,
        )
        return result

    def _format_action(self, action):
        arr = np.asarray(action, dtype=np.float32)
        single_shape = tuple(int(v) for v in getattr(self.action_space, "shape", ()))
        if single_shape and tuple(arr.shape) == (self._num_envs, *single_shape):
            pass
        elif single_shape and tuple(arr.shape) != single_shape:
            arr = np.reshape(arr, single_shape)
        if self._num_envs == 1 and single_shape and tuple(arr.shape) == single_shape:
            arr = np.expand_dims(arr, axis=0)
        try:
            import torch

            device = getattr(self._env, "device", None)
            return torch.as_tensor(arr, dtype=torch.float32, device=device)
        except Exception:
            return arr


class IsaacLabVectorEnvAdapter(_IsaacLabAdapterBase):
    def __init__(self, env: Any, *, num_envs: int) -> None:
        self._init_adapter(env, num_envs=int(num_envs))

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset_batch(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        return _flatten_obs_batch(obs, num_envs=self._num_envs), info

    def step_batch(self, actions):
        step_out = self._env.step(self._format_actions(actions))
        if len(step_out) != 5:
            raise ValueError(f"Unsupported Isaac Lab step return arity: {len(step_out)}")
        obs, reward, terminated, truncated, info = step_out
        return (
            _flatten_obs_batch(obs, num_envs=self._num_envs),
            np.ravel(_as_numpy(reward).astype(np.float32, copy=False)),
            np.ravel(_as_numpy(terminated).astype(bool, copy=False)),
            np.ravel(_as_numpy(truncated).astype(bool, copy=False)),
            info,
        )

    def _format_actions(self, actions):
        arr = np.asarray(actions, dtype=np.float32)
        single_shape = tuple(int(v) for v in getattr(self.action_space, "shape", ()))
        target_shape = (self._num_envs, *single_shape)
        if single_shape and tuple(arr.shape) != target_shape:
            arr = np.reshape(arr, target_shape)
        try:
            import torch

            device = getattr(self._env, "device", None)
            return torch.as_tensor(arr, dtype=torch.float32, device=device)
        except Exception:
            return arr
