from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from common.video_rollout import rollout_episode


def select_video_episode_indices(
    episode_returns: list[float],
    *,
    selection: str,
    num_video_episodes: int,
    base_seed: int,
) -> list[int]:
    if num_video_episodes <= 0:
        return []
    count = min(int(num_video_episodes), len(episode_returns))
    if selection == "first":
        return list(range(count))
    if selection == "random":
        rng = np.random.default_rng(base_seed)
        return list(rng.choice(len(episode_returns), size=count, replace=False))
    ranked = sorted(range(len(episode_returns)), key=lambda i: episode_returns[i], reverse=True)
    return ranked[:count]


def _is_headless_video_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "display environment variable is missing",
        "opengl platform library has not been loaded",
        "valid opengl context has not been created",
        "mjr_makecontext",
        "mjrcontext",
        "glfw",
        "x11",
        "headless",
    )
    return ("fatalerror" in name) or any(n in msg for n in needles)


def _video_gl_candidates() -> list[str | None]:
    user_gl = os.environ.get("MUJOCO_GL")
    if user_gl is not None and str(user_gl).strip() != "":
        return [str(user_gl).strip()]
    if not sys.platform.startswith("linux"):
        return [None]
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        return [None]
    return ["egl", "osmesa"]


@contextmanager
def _temporary_mujoco_gl(value: str | None):
    prev = os.environ.get("MUJOCO_GL")
    try:
        if value is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = str(value)
        yield
    finally:
        if prev is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = prev


_AUTO_GL = object()


def _render_video_episode(
    env_conf: Any,
    policy: Any,
    *,
    seed: int,
    video_dir: Path,
    video_prefix: str,
    gl_backend: str | None,
) -> None:
    with _temporary_mujoco_gl(gl_backend):
        rollout_episode(
            env_conf,
            policy,
            seed=seed,
            render_video=True,
            video_dir=video_dir,
            video_prefix=video_prefix,
        )


def render_policy_videos(
    env_conf: Any,
    policy: Any,
    *,
    video_dir: Path,
    video_prefix: str,
    num_episodes: int,
    num_video_episodes: int,
    episode_selection: str,
    seed_base: int,
) -> None:
    selection = str(episode_selection).lower()
    if selection not in ("best", "first", "random"):
        raise ValueError("episode_selection must be one of: best, first, random")

    video_dir.mkdir(parents=True, exist_ok=True)
    episode_returns = [
        rollout_episode(
            env_conf,
            policy,
            seed=seed_base + episode_idx,
            render_video=False,
            video_dir=None,
            video_prefix=video_prefix,
        )
        for episode_idx in range(num_episodes)
    ]
    selected_indices = select_video_episode_indices(
        episode_returns,
        selection=selection,
        num_video_episodes=num_video_episodes,
        base_seed=seed_base,
    )
    print(
        f"[video] dir={video_dir} episodes={num_episodes} videos={len(selected_indices)} select={selection}",
        flush=True,
    )
    preferred_gl: str | None | object = _AUTO_GL
    gl_candidates = _video_gl_candidates()
    for episode_idx in selected_indices:
        backends = [preferred_gl] if preferred_gl is not _AUTO_GL else gl_candidates
        last_headless_exc: Exception | None = None
        rendered = False
        for gl_backend in backends:
            try:
                _render_video_episode(
                    env_conf,
                    policy,
                    seed=seed_base + episode_idx,
                    video_dir=video_dir,
                    video_prefix=f"{video_prefix}_ep{episode_idx:03d}",
                    gl_backend=gl_backend,
                )
                if preferred_gl is _AUTO_GL:
                    preferred_gl = gl_backend
                    if gl_backend is not None:
                        print(f"[video] using MUJOCO_GL={gl_backend}", flush=True)
                rendered = True
                break
            except Exception as exc:
                if _is_headless_video_error(exc):
                    last_headless_exc = exc
                    continue
                raise
        if not rendered:
            print(
                f"[video] skipping video capture: headless/OpenGL context unavailable ({last_headless_exc})",
                flush=True,
            )
            return
