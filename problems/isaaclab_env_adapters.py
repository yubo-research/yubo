from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np


ISAACLAB_ENV_PREFIX = "isaaclab:"
DEFAULT_ISAACLAB_MAX_STEPS = 1000


def is_isaaclab_env_tag(env_tag: str) -> bool:
    return str(env_tag).startswith(ISAACLAB_ENV_PREFIX)


def parse_isaaclab_task_id(env_tag: str) -> str:
    tag = str(env_tag)
    if not is_isaaclab_env_tag(tag):
        raise ValueError(f"Unsupported Isaac Lab env_tag: {tag!r}. Expected prefix {ISAACLAB_ENV_PREFIX!r}.")
    task_id = tag.split(":", 1)[1].strip()
    if not task_id:
        raise ValueError(f"Isaac Lab env_tag {tag!r} is missing a task id, e.g. 'isaaclab:Isaac-Cartpole-v0'.")
    return task_id


@dataclass
class IsaacLabSession:
    app: Any
    gym: Any


_SESSION: IsaacLabSession | None = None


def _import_isaac_app_launcher():
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise ImportError(
            "Isaac Lab is not importable. Run admin/setup-hyperscalees.sh on the CUDA remote, then use an env_tag like 'isaaclab:Isaac-Cartpole-v0'."
        ) from exc
    return AppLauncher


def get_isaaclab_session(*, headless: bool = True, launcher_kwargs: dict[str, Any] | None = None) -> IsaacLabSession:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    kwargs = {"headless": bool(headless)}
    if launcher_kwargs:
        kwargs.update(dict(launcher_kwargs))

    AppLauncher = _import_isaac_app_launcher()
    try:
        launcher = AppLauncher(**kwargs)
    except TypeError:
        launcher = AppLauncher(kwargs)

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    _SESSION = IsaacLabSession(app=launcher.app, gym=gym)
    return _SESSION


def list_isaaclab_tasks(*, keyword: str | None = None, headless: bool = True) -> list[str]:
    session = get_isaaclab_session(headless=headless)
    needle = None if keyword is None else str(keyword).lower()
    tasks: list[str] = []
    for spec in session.gym.registry.values():
        task_id = getattr(spec, "id", "")
        if not task_id or "Isaac" not in str(task_id):
            continue
        if needle is not None and needle not in str(task_id).lower():
            continue
        tasks.append(str(task_id))
    return sorted(set(tasks))


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


def _single_action_space(env: Any):
    return getattr(env, "single_action_space", None) or getattr(env, "action_space")


def _single_observation_space(env: Any):
    return getattr(env, "single_observation_space", None) or getattr(env, "observation_space")


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


class IsaacLabGymEnvAdapter:
    def __init__(self, env: Any, *, num_envs: int = 1) -> None:
        self._env = env
        self._num_envs = int(num_envs)
        self.observation_space = _flat_box_from_space(_single_observation_space(env))
        self.action_space = _single_action_space(env)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        return _flatten_obs(obs), info

    def step(self, action):
        step_out = self._env.step(self._format_action(action))
        if len(step_out) != 5:
            raise ValueError(f"Unsupported Isaac Lab step return arity: {len(step_out)}")
        obs, reward, terminated, truncated, info = step_out
        result = (_flatten_obs(obs), _scalar_float(reward), _scalar_bool(terminated), _scalar_bool(truncated), info)
        return result

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

    def _format_action(self, action):
        arr = np.asarray(action, dtype=np.float32)
        if hasattr(self.action_space, "shape") and tuple(self.action_space.shape) != tuple(arr.shape):
            arr = np.reshape(arr, tuple(int(v) for v in self.action_space.shape))
        if self._num_envs == 1 and arr.ndim >= 1:
            arr = np.expand_dims(arr, axis=0)
        try:
            import torch

            device = getattr(self._env, "device", None)
            return torch.as_tensor(arr, dtype=torch.float32, device=device)
        except Exception:
            return arr


def _parse_env_cfg(task_id: str, *, num_envs: int, device: str | None):
    from isaaclab_tasks.utils import parse_env_cfg

    kwargs: dict[str, Any] = {"num_envs": int(num_envs)}
    if device is not None:
        kwargs["device"] = str(device)
    try:
        return parse_env_cfg(task_id, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return parse_env_cfg(task_id, **kwargs)


def make_isaaclab_env(
    env_tag: str,
    *,
    headless: bool = True,
    num_envs: int = 1,
    device: str | None = None,
    render_mode: str | None = None,
    launcher_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    task_id = parse_isaaclab_task_id(env_tag)
    session = get_isaaclab_session(headless=headless, launcher_kwargs=launcher_kwargs)

    make_kwargs = dict(kwargs)
    if "cfg" not in make_kwargs:
        make_kwargs["cfg"] = _parse_env_cfg(task_id, num_envs=int(num_envs), device=device)
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    env = session.gym.make(task_id, **make_kwargs)
    return IsaacLabGymEnvAdapter(env, num_envs=int(num_envs))


def resolve_isaaclab_env_spaces(env_tag: str):
    env = make_isaaclab_env(env_tag, num_envs=1)
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List Isaac Lab Gymnasium task ids.")
    parser.add_argument("--keyword", default=None, help="Optional case-insensitive substring filter.")
    parser.add_argument("--no-headless", action="store_true", help="Launch Isaac Lab with headless=False.")
    args = parser.parse_args(argv)
    for task_id in list_isaaclab_tasks(keyword=args.keyword, headless=not args.no_headless):
        print(task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
