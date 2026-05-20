from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from problems.isaaclab_config import ISAACLAB_DEFAULT_KIT_ARGS, disable_command_debug_visualizers

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
    headless: bool = True
    launcher_kwargs: dict[str, Any] | None = None
    render_available: bool = False
    render_error: str | None = None


_SESSION: IsaacLabSession | None = None
_SPACE_CACHE: dict[str, tuple[Any, Any]] = {}
MAX_ISAACLAB_SETUP_REPLAY_CHARS = 20_000


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _tail_text(text: str, *, max_chars: int = MAX_ISAACLAB_SETUP_REPLAY_CHARS) -> str:
    if len(text) <= int(max_chars):
        return text
    return f"[isaaclab setup output truncated to last {max_chars} chars]\n{text[-int(max_chars) :]}"


def _replay_captured_setup_output(captured) -> None:
    captured.flush()
    captured.seek(0)
    text = captured.read()
    if text:
        print(_tail_text(text), end="", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _capture_isaaclab_setup_output():
    _flush_standard_streams()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    failed = False
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as captured:
        os.dup2(captured.fileno(), 1)
        os.dup2(captured.fileno(), 2)
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                yield
        except Exception:
            failed = True
            raise
        finally:
            captured.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            if failed:
                _replay_captured_setup_output(captured)


def _import_isaac_app_launcher():
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise ImportError(
            "Isaac Lab is not importable. Run admin/setup-hyperscalees.sh on the CUDA remote, then use an env_tag like 'isaaclab:Isaac-Cartpole-v0'."
        ) from exc
    return AppLauncher


def _copy_launcher_kwargs(launcher_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    return {} if launcher_kwargs is None else dict(launcher_kwargs)


def isaaclab_default_launcher_kwargs() -> dict[str, Any]:
    return {"kit_args": ISAACLAB_DEFAULT_KIT_ARGS}


def _render_probe() -> tuple[bool, str | None]:
    try:
        import_module("omni.replicator.core")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def isaaclab_rendering_available() -> tuple[bool, str | None]:
    if _SESSION is None:
        return _render_probe()
    return bool(_SESSION.render_available), _SESSION.render_error


def _session_can_reuse(session: IsaacLabSession, *, headless: bool, launcher_kwargs: dict[str, Any]) -> tuple[bool, str | None]:
    if bool(session.headless) and not bool(headless):
        return False, "existing IsaacLab app is headless; it cannot be upgraded to headless=False in the same process"
    existing_kwargs = dict(session.launcher_kwargs or {})
    if launcher_kwargs and existing_kwargs != dict(launcher_kwargs):
        return False, f"existing IsaacLab app launcher_kwargs={existing_kwargs!r}, requested launcher_kwargs={launcher_kwargs!r}"
    return True, None


def get_isaaclab_session(*, headless: bool = True, launcher_kwargs: dict[str, Any] | None = None) -> IsaacLabSession:
    global _SESSION
    if _SESSION is not None:
        if launcher_kwargs is None:
            return _SESSION
        launcher_kwargs = _copy_launcher_kwargs(launcher_kwargs)
        ok, reason = _session_can_reuse(_SESSION, headless=bool(headless), launcher_kwargs=launcher_kwargs)
        if not ok:
            raise RuntimeError(f"IsaacLab app is already running with incompatible settings: {reason}.")
        return _SESSION

    launcher_kwargs = _copy_launcher_kwargs(launcher_kwargs)
    if not launcher_kwargs:
        launcher_kwargs = isaaclab_default_launcher_kwargs()

    effective_launcher_kwargs = dict(launcher_kwargs)
    kit_args = str(effective_launcher_kwargs.get("kit_args", "")).strip()
    effective_launcher_kwargs["kit_args"] = kit_args

    kwargs = {"headless": bool(headless)}
    kwargs.update(effective_launcher_kwargs)

    AppLauncher = _import_isaac_app_launcher()
    with _capture_isaaclab_setup_output():
        try:
            launcher = AppLauncher(**kwargs)
        except TypeError:
            launcher = AppLauncher(kwargs)

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401

    if "omni.replicator.core" in kit_args:
        render_available, render_error = _render_probe()
    else:
        render_available, render_error = False, "omni.replicator.core not enabled"
    _SESSION = IsaacLabSession(
        app=launcher.app,
        gym=gym,
        headless=bool(headless),
        launcher_kwargs=launcher_kwargs,
        render_available=bool(render_available),
        render_error=render_error,
    )
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


class IsaacLabGymEnvAdapter:
    def __init__(self, env: Any, *, num_envs: int = 1) -> None:
        self._env = env
        self._num_envs = int(num_envs)
        self.observation_space, self.action_space = _adapter_spaces(env, num_envs=self._num_envs)

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

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

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


class IsaacLabVectorEnvAdapter:
    def __init__(self, env: Any, *, num_envs: int) -> None:
        self._env = env
        self._num_envs = int(num_envs)
        self.observation_space, self.action_space = _adapter_spaces(env, num_envs=self._num_envs)

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

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

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
    batched = bool(make_kwargs.pop("batched", False))
    seed = make_kwargs.pop("seed", None)
    with _capture_isaaclab_setup_output():
        if "cfg" not in make_kwargs:
            cfg = _parse_env_cfg(task_id, num_envs=int(num_envs), device=device)
            make_kwargs["cfg"] = cfg
        if seed is not None and hasattr(make_kwargs["cfg"], "seed"):
            try:
                make_kwargs["cfg"].seed = int(seed)
            except Exception:
                pass
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode

        disable_command_debug_visualizers(make_kwargs["cfg"])
        env = session.gym.make(task_id, **make_kwargs)
    adapter = IsaacLabVectorEnvAdapter(env, num_envs=int(num_envs)) if batched else IsaacLabGymEnvAdapter(env, num_envs=int(num_envs))
    _SPACE_CACHE[str(env_tag)] = (adapter.observation_space, adapter.action_space)
    return adapter


def resolve_isaaclab_env_spaces(env_tag: str):
    cached = _SPACE_CACHE.get(str(env_tag))
    if cached is not None:
        return cached
    env = make_isaaclab_env(env_tag, num_envs=1)
    try:
        spaces = (env.observation_space, env.action_space)
        _SPACE_CACHE[str(env_tag)] = spaces
        return spaces
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List Isaac Lab Gymnasium task ids.")
    parser.add_argument("--keyword", default=None, help="Optional case-insensitive substring filter.")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Launch Isaac Lab with headless=False.",
    )
    args = parser.parse_args(argv)
    for task_id in list_isaaclab_tasks(keyword=args.keyword, headless=not args.no_headless):
        print(task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
