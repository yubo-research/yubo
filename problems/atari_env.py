"""Atari environments via Gymnasium + ALE. Standard DQN-style preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

# Register ALE envs with Gymnasium
try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:
    pass  # ale_py not installed; gym.make will fail with clear error

ATARI_FRAME_SIZE = 84
ATARI_FRAME_STACK = 4
ATARI_MAX_EPISODE_STEPS = 108000  # 30 min at 60fps, 4 frame skip -> ~27k steps


@dataclass(frozen=True)
class AtariPreprocessOptions:
    """Options for Atari preprocessing. Replaces 4 boolean flags."""

    terminal_on_life_loss: bool = False
    grayscale_obs: bool = True
    grayscale_newaxis: bool = True
    scale_obs: bool = False


def _parse_atari_tag(tag: str) -> str:
    """Parse atari:Pong or atari:Pong:agent57 -> ALE/Pong-v5."""
    if tag.startswith("atari:"):
        parts = tag.split(":", 1)[1].strip().split(":")
        game = parts[0].split("-")[0]
    elif tag.startswith("ALE/"):
        return tag if "-v" in tag else f"{tag}-v5"
    else:
        raise ValueError(f"Expected atari:Game or ALE/Game-v5, got: {tag}")
    return f"ALE/{game}-v5"


def make_atari_env(
    env_id: str,
    *,
    render_mode: str | None = None,
    frameskip: int = 1,
    noop_max: int = 30,
    frame_skip: int = 4,
    screen_size: int = ATARI_FRAME_SIZE,
    preprocess: AtariPreprocessOptions | None = None,
    max_episode_steps: int = ATARI_MAX_EPISODE_STEPS,
    **kwargs: Any,
) -> gym.Env:
    """Create Atari env with DQN-style preprocessing.

    Pipeline: base env -> AtariPreprocessing -> FrameStack -> TimeLimit.

    Observation: (84, 84, 4) grayscale stacked frames, uint8 [0,255] or float [0,1] if scale_obs.
    Action: Discrete.
    """
    make_kwargs = {"frameskip": frameskip}
    if render_mode:
        make_kwargs["render_mode"] = render_mode
    base = gym.make(env_id, **make_kwargs)

    opts = preprocess if preprocess is not None else AtariPreprocessOptions()
    base = gym.wrappers.AtariPreprocessing(
        base,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=opts.terminal_on_life_loss,
        grayscale_obs=opts.grayscale_obs,
        grayscale_newaxis=opts.grayscale_newaxis,
        scale_obs=opts.scale_obs,
    )
    base = gym.wrappers.FrameStackObservation(base, stack_size=ATARI_FRAME_STACK)
    base = gym.wrappers.TimeLimit(base, max_episode_steps=max_episode_steps)
    return base


def make(env_id: str, *, render_mode: str | None = None, **kwargs: Any) -> gym.Env:
    """Create Atari env. env_id can be atari:Pong or ALE/Pong-v5."""
    resolved = _parse_atari_tag(env_id) if "atari:" in env_id or env_id.startswith("ALE/") else env_id
    return make_atari_env(resolved, render_mode=render_mode, **kwargs)
