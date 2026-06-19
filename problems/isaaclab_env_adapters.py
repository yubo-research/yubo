from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from problems.isaaclab_config import ISAACLAB_DEFAULT_KIT_ARGS, ISAACLAB_VIDEO_KIT_ARGS
from problems.isaaclab_gym_adapters import (
    IsaacLabGymEnvAdapter,
    IsaacLabVectorEnvAdapter,
)
from problems.isaaclab_replicator import patch_replicator_seed_without_graph

ISAACLAB_ENV_PREFIX = "isaaclab:"
DEFAULT_ISAACLAB_MAX_STEPS = 1000
MAX_ISAACLAB_SETUP_REPLAY_CHARS = 20_000


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
            with (
                contextlib.redirect_stdout(captured),
                contextlib.redirect_stderr(captured),
            ):
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
            "Isaac Lab is not importable. Run the Pixi setup task on the CUDA remote, then use an env_tag like 'isaaclab:Isaac-Cartpole-v0'."
        ) from exc
    return AppLauncher


def _copy_launcher_kwargs(launcher_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    return {} if launcher_kwargs is None else dict(launcher_kwargs)


def _nvidia_visible_devices_disabled() -> bool:
    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if visible is None:
        return False
    return visible.strip().lower() in {"", "none", "void"}


def _torch_cuda_usable() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return False
    try:
        torch.zeros(1, device="cuda:0")
        return True
    except Exception:
        return False


def resolve_isaaclab_sim_device(requested: str | None = None) -> str:
    """Pick Isaac Lab simulation device (must match PhysX/Warp tensor placement)."""
    override = os.environ.get("YUBO_ISAACLAB_DEVICE", "").strip()
    if override:
        return str(override)

    mode = "auto" if requested in (None, "", "auto") else str(requested).strip().lower()
    if mode == "cpu":
        return "cpu"
    if mode in {"cuda", "gpu"}:
        if _torch_cuda_usable() and not _nvidia_visible_devices_disabled():
            return "cuda:0"
        print(
            "ISAACLAB_WARN: cuda requested for Isaac Lab simulation but no usable CUDA GPU was found; using cpu",
            flush=True,
        )
        return "cpu"
    if mode.startswith("cuda:"):
        if _torch_cuda_usable() and not _nvidia_visible_devices_disabled():
            return mode
        print(
            f"ISAACLAB_WARN: {mode!r} requested for Isaac Lab simulation but no usable CUDA GPU was found; using cpu",
            flush=True,
        )
        return "cpu"
    if _torch_cuda_usable() and not _nvidia_visible_devices_disabled():
        return "cuda:0"
    return "cpu"


def isaaclab_default_launcher_kwargs() -> dict[str, Any]:
    return {"kit_args": ISAACLAB_DEFAULT_KIT_ARGS}


def isaaclab_video_launcher_kwargs() -> dict[str, Any]:
    return {"kit_args": ISAACLAB_VIDEO_KIT_ARGS, "video": True}


def _render_probe() -> tuple[bool, str | None]:
    try:
        import_module("omni.kit.viewport.utility")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def isaaclab_rendering_available() -> tuple[bool, str | None]:
    if _SESSION is None:
        return _render_probe()
    return bool(_SESSION.render_available), _SESSION.render_error


def _session_can_reuse(session: IsaacLabSession, *, headless: bool, launcher_kwargs: dict[str, Any]) -> tuple[bool, str | None]:
    if bool(session.headless) and not bool(headless):
        return (
            False,
            "existing IsaacLab app is headless; it cannot be upgraded to headless=False in the same process",
        )
    existing_kwargs = dict(session.launcher_kwargs or {})
    if launcher_kwargs and existing_kwargs != dict(launcher_kwargs):
        return (
            False,
            f"existing IsaacLab app launcher_kwargs={existing_kwargs!r}, requested launcher_kwargs={launcher_kwargs!r}",
        )
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

    render_available, render_error = _session_render_state(kit_args)
    _SESSION = IsaacLabSession(
        app=launcher.app,
        gym=gym,
        headless=bool(headless),
        launcher_kwargs=launcher_kwargs,
        render_available=bool(render_available),
        render_error=render_error,
    )
    return _SESSION


def _session_render_state(kit_args: str) -> tuple[bool, str | None]:
    if "omni.kit.viewport.utility" in kit_args:
        return _render_probe()
    if "omni.replicator.core" in kit_args:
        patch_replicator_seed_without_graph()
        return _render_probe()
    return False, "omni.kit.viewport.utility not enabled"


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


def _parse_env_cfg(task_id: str, *, num_envs: int, device: str | None):
    from isaaclab_tasks.utils import parse_env_cfg

    kwargs: dict[str, Any] = {
        "num_envs": int(num_envs),
        "device": resolve_isaaclab_sim_device(device),
    }
    try:
        return parse_env_cfg(task_id, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return parse_env_cfg(task_id, **kwargs)


def make_raw_isaaclab_env(
    env_tag: str,
    *,
    headless: bool = True,
    num_envs: int = 1,
    device: str | None = None,
    render_mode: str | None = None,
    launcher_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    from problems.isaaclab_env_make import (
        make_raw_isaaclab_env as _make_raw_isaaclab_env,
    )

    return _make_raw_isaaclab_env(
        sys.modules[__name__],
        env_tag,
        headless=headless,
        num_envs=int(num_envs),
        device=device,
        render_mode=render_mode,
        launcher_kwargs=launcher_kwargs,
        **kwargs,
    )


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
    make_kwargs = dict(kwargs)
    batched = bool(make_kwargs.pop("batched", False))
    env = make_raw_isaaclab_env(
        env_tag,
        headless=headless,
        num_envs=int(num_envs),
        device=device,
        render_mode=render_mode,
        launcher_kwargs=launcher_kwargs,
        **make_kwargs,
    )
    adapter = IsaacLabVectorEnvAdapter(env, num_envs=int(num_envs)) if batched else IsaacLabGymEnvAdapter(env, num_envs=int(num_envs))
    _SPACE_CACHE[str(env_tag)] = (adapter.observation_space, adapter.action_space)
    return adapter


def resolve_isaaclab_env_spaces(env_tag: str, *, launcher_kwargs: dict[str, Any] | None = None):
    cached = _SPACE_CACHE.get(str(env_tag))
    if cached is not None:
        if launcher_kwargs is not None:
            get_isaaclab_session(launcher_kwargs=launcher_kwargs)
        return cached
    env = make_isaaclab_env(
        env_tag,
        num_envs=1,
        launcher_kwargs=launcher_kwargs,
        device=resolve_isaaclab_sim_device(None),
    )
    try:
        spaces = (env.observation_space, env.action_space)
        _SPACE_CACHE[str(env_tag)] = spaces
        return spaces
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

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
