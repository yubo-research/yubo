from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rl.core.continuous_actions import normalize_action_bounds
from rl.core.env_conf import resolve_noise_seed_0, resolve_problem_seed


def _maybe_register_atari_dm_backends(env_tag: str) -> None:
    if not str(env_tag).startswith(("atari:", "ALE/", "dm:", "dm_control/")):
        return


@dataclass(frozen=True)
class ContinuousGymEnvSetup:
    env_conf: Any
    problem_seed: int
    noise_seed_0: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None


def build_continuous_gym_env_setup(
    *,
    env_tag: str,
    seed: int,
    problem_seed: int | None,
    noise_seed_0: int | None,
    from_pixels: bool,
    pixels_only: bool,
    get_env_conf_fn: Callable[..., Any],
    obs_scale_from_env_fn: Callable[[Any], tuple[np.ndarray | None, np.ndarray | None]],
) -> ContinuousGymEnvSetup:
    resolved_problem_seed = resolve_problem_seed(seed=int(seed), problem_seed=problem_seed)
    resolved_noise_seed_0 = resolve_noise_seed_0(problem_seed=int(resolved_problem_seed), noise_seed_0=noise_seed_0)
    _maybe_register_atari_dm_backends(str(env_tag))
    env_conf = get_env_conf_fn(
        str(env_tag),
        problem_seed=int(resolved_problem_seed),
        noise_seed_0=int(resolved_noise_seed_0),
        from_pixels=bool(from_pixels),
        pixels_only=bool(pixels_only),
    )
    env_conf.ensure_spaces()
    if not hasattr(env_conf.action_space, "shape") or not hasattr(env_conf.action_space, "low"):
        raise ValueError("SAC expects a continuous Box action space.")
    action_shape = tuple((int(v) for v in env_conf.action_space.shape))
    act_dim = int(np.prod(action_shape)) if action_shape else 1
    action_low, action_high = normalize_action_bounds(np.asarray(env_conf.action_space.low), np.asarray(env_conf.action_space.high), dim=act_dim)
    obs_lb, obs_width = obs_scale_from_env_fn(env_conf)
    if obs_lb is not None and obs_width is not None:
        obs_lb = np.asarray(obs_lb, dtype=np.float32).reshape(-1)
        obs_width = np.asarray(obs_width, dtype=np.float32).reshape(-1)
    return ContinuousGymEnvSetup(
        env_conf=env_conf,
        problem_seed=int(resolved_problem_seed),
        noise_seed_0=int(resolved_noise_seed_0),
        act_dim=act_dim,
        action_low=np.asarray(action_low, dtype=np.float32),
        action_high=np.asarray(action_high, dtype=np.float32),
        obs_lb=obs_lb,
        obs_width=obs_width,
    )
