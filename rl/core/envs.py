from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from common.experiment_seeds import (
    global_seed_for_run,
    noise_seed_0_from_problem_seed,
    problem_seed_from_rep_index,
)
from common.obs_mode import normalize_obs_mode
from rl.core.continuous_actions import normalize_action_bounds


def problem_seed(*, seed: int, problem_seed: int | None) -> int:
    return int(problem_seed) if problem_seed is not None else problem_seed_from_rep_index(max(0, int(seed)))


def noise_seed_0(*, problem_seed: int, noise_seed_0: int | None) -> int:
    return int(noise_seed_0) if noise_seed_0 is not None else noise_seed_0_from_problem_seed(problem_seed)


@dataclass(frozen=True)
class ResolvedSeeds:
    problem_seed: int
    noise_seed_0: int


@dataclass(frozen=True)
class SeededEnvConf:
    env_conf: Any
    problem_seed: int
    noise_seed_0: int


@dataclass(frozen=True)
class ContinuousEnvSetup:
    env_conf: Any
    problem_seed: int
    noise_seed_0: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None


def seeds(*, seed: int, problem_seed: int | None, noise_seed_0: int | None) -> ResolvedSeeds:
    pseed = int(problem_seed) if problem_seed is not None else problem_seed_from_rep_index(max(0, int(seed)))
    nseed = int(noise_seed_0) if noise_seed_0 is not None else noise_seed_0_from_problem_seed(int(pseed))
    return ResolvedSeeds(problem_seed=int(pseed), noise_seed_0=int(nseed))


def seeded_conf(
    *,
    env_tag: str,
    problem_seed: int,
    noise_seed_0: int,
    obs_mode: str,
    get_env_conf_fn: Callable[..., Any],
) -> SeededEnvConf:
    env_conf = get_env_conf_fn(
        str(env_tag),
        problem_seed=int(problem_seed),
        noise_seed_0=int(noise_seed_0),
        obs_mode=normalize_obs_mode(obs_mode),
    )
    return SeededEnvConf(
        env_conf=env_conf,
        problem_seed=int(problem_seed),
        noise_seed_0=int(noise_seed_0),
    )


def conf_for_run(
    *,
    env_tag: str,
    seed: int,
    problem_seed: int | None,
    noise_seed_0: int | None,
    obs_mode: str,
    get_env_conf_fn: Callable[..., Any],
) -> SeededEnvConf:
    out = seeds(seed=int(seed), problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    return seeded_conf(
        env_tag=str(env_tag),
        problem_seed=int(out.problem_seed),
        noise_seed_0=int(out.noise_seed_0),
        obs_mode=str(obs_mode),
        get_env_conf_fn=get_env_conf_fn,
    )


def build_continuous_env_setup(
    *,
    env_tag: str,
    seed: int,
    problem_seed: int | None,
    noise_seed_0: int | None,
    obs_mode: str,
    get_env_conf_fn: Callable[..., Any],
    obs_scale_from_env_fn: Callable[[Any], tuple[np.ndarray | None, np.ndarray | None]],
) -> ContinuousEnvSetup:
    out = conf_for_run(
        env_tag=str(env_tag),
        seed=int(seed),
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        obs_mode=str(obs_mode),
        get_env_conf_fn=get_env_conf_fn,
    )
    env_conf = out.env_conf
    env_conf.ensure_spaces()
    if not hasattr(env_conf.action_space, "shape") or not hasattr(env_conf.action_space, "low"):
        raise ValueError("SAC expects a continuous Box action space.")
    action_shape = tuple(int(v) for v in env_conf.action_space.shape)
    act_dim = int(np.prod(action_shape)) if action_shape else 1
    action_low, action_high = normalize_action_bounds(
        np.asarray(env_conf.action_space.low),
        np.asarray(env_conf.action_space.high),
        dim=act_dim,
    )
    obs_lb, obs_width = obs_scale_from_env_fn(env_conf)
    if obs_lb is not None and obs_width is not None:
        obs_lb = np.asarray(obs_lb, dtype=np.float32).reshape(-1)
        obs_width = np.asarray(obs_width, dtype=np.float32).reshape(-1)
    return ContinuousEnvSetup(
        env_conf=env_conf,
        problem_seed=int(out.problem_seed),
        noise_seed_0=int(out.noise_seed_0),
        act_dim=act_dim,
        action_low=np.asarray(action_low, dtype=np.float32),
        action_high=np.asarray(action_high, dtype=np.float32),
        obs_lb=obs_lb,
        obs_width=obs_width,
    )


__all__ = [
    "ContinuousEnvSetup",
    "ResolvedSeeds",
    "SeededEnvConf",
    "build_continuous_env_setup",
    "conf_for_run",
    "global_seed_for_run",
    "noise_seed_0",
    "problem_seed",
    "seeded_conf",
    "seeds",
]
