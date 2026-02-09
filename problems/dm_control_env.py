from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class _ParsedEnv:
    domain: str
    task: str
    version: str | None


def _parse_env_name(env_name: str) -> _ParsedEnv:
    if not env_name.startswith("dm_control/"):
        raise ValueError(f"Expected dm_control env name, got: {env_name}")
    name = env_name.split("/", 1)[1]
    version = None
    if name.endswith("-v0") or name.endswith("-v1"):
        name, version = name.rsplit("-", 1)
        version = version.lower()
    if "-" not in name:
        raise ValueError(f"Expected dm_control/<domain>-<task>-v0, got: {env_name}")
    domain, task = name.split("-", 1)
    return _ParsedEnv(domain=domain, task=task, version=version)


def _spec_bounds(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    minimum = getattr(spec, "minimum", None)
    maximum = getattr(spec, "maximum", None)
    shape = spec.shape
    low = np.full(shape, -np.inf, dtype=np.float32)
    high = np.full(shape, np.inf, dtype=np.float32)
    if minimum is not None:
        low = np.asarray(minimum, dtype=np.float32)
    if maximum is not None:
        high = np.asarray(maximum, dtype=np.float32)
    return low, high


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [np.ravel(np.asarray(obs[k], dtype=np.float32)) for k in sorted(obs)]
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)
    return np.ravel(np.asarray(obs, dtype=np.float32))


def _spec_to_space(spec: Any):
    if isinstance(spec, dict):
        lows, highs = [], []
        for key in sorted(spec):
            low, high = _spec_bounds(spec[key])
            lows.append(np.ravel(low))
            highs.append(np.ravel(high))
        low = np.concatenate(lows, axis=0)
        high = np.concatenate(highs, axis=0)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
    low, high = _spec_bounds(spec)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def _get_offscreen_size(env) -> tuple[int | None, int | None]:
    try:
        vis = env.physics.model.vis
    except Exception:
        return None, None
    global_cfg = None
    if hasattr(vis, "global_"):
        global_cfg = vis.global_
    elif hasattr(vis, "global"):
        global_cfg = getattr(vis, "global")
    if global_cfg is None:
        return None, None
    return getattr(global_cfg, "offwidth", None), getattr(global_cfg, "offheight", None)


def _camera_id(env, name: str) -> int:
    model = env.physics.model
    try:
        return int(model.name2id(name, "camera"))
    except Exception:
        return -1


DEFAULT_RENDER_WIDTH = 1280
DEFAULT_RENDER_HEIGHT = 720
DEFAULT_RENDER_CAMERA = "side"


class DMControlEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain: str,
        task: str,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self._domain = domain
        self._task = task
        self._seed = seed
        self.render_mode = render_mode
        self._render_width = DEFAULT_RENDER_WIDTH
        self._render_height = DEFAULT_RENDER_HEIGHT
        self._env = self._load_env(seed)
        self._obs_spec = self._env.observation_spec()
        self._action_spec = self._env.action_spec()
        self.observation_space = _spec_to_space(self._obs_spec)
        self.action_space = _spec_to_space(self._action_spec)

    def _load_env(self, seed: int | None):
        try:
            from dm_control import suite
        except Exception as exc:
            raise ImportError("dm_control is not installed. Install it to use dm_control environments.") from exc
        task_kwargs = {"random": seed} if seed is not None else None
        return suite.load(self._domain, self._task, task_kwargs=task_kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and seed != self._seed:
            self._seed = seed
            self._env = self._load_env(seed)
        time_step = self._env.reset()
        obs = _flatten_obs(time_step.observation)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=self._action_spec.dtype)
        time_step = self._env.step(action)
        obs = _flatten_obs(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = bool(time_step.last())
        truncated = False
        step_result = (obs, reward, terminated, truncated, {"discount": time_step.discount})
        return step_result

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        width = int(self._render_width)
        height = int(self._render_height)
        offwidth, offheight = _get_offscreen_size(self._env)
        if offwidth is not None:
            width = min(width, int(offwidth))
        if offheight is not None:
            height = min(height, int(offheight))
        kwargs = {"width": width, "height": height}
        camera_id = _camera_id(self._env, DEFAULT_RENDER_CAMERA)
        if camera_id < 0:
            raise ValueError(f"Camera '{DEFAULT_RENDER_CAMERA}' not found in dm_control model.")
        return self._env.physics.render(**kwargs, camera_id=camera_id)

    def close(self):
        try:
            self._env.close()
        except Exception:
            return


def make(
    env_name: str,
    *,
    render_mode: str | None = None,
):
    parsed = _parse_env_name(env_name)
    return DMControlEnv(parsed.domain, parsed.task, render_mode=render_mode)
