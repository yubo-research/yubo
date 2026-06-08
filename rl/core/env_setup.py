from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from common.experiment_seeds import resolve_noise_seed_0, resolve_problem_seed
from problems.env_conf_backends import maybe_register_atari_dm_backends
from rl.core.continuous_actions import normalize_action_bounds
from rl.core.runtime import obs_scale_from_env


@dataclass(frozen=True)
class EnvSetup:
    env_conf: Any
    problem_seed: int
    noise_seed_0: int
    # Optional continuous space info
    act_dim: int | None = None
    action_low: np.ndarray | None = None
    action_high: np.ndarray | None = None
    obs_lb: np.ndarray | None = None
    obs_width: np.ndarray | None = None


def build_env_setup(
    *,
    env_tag: str,
    seed: int,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    from_pixels: bool = False,
    pixels_only: bool = True,
    get_env_conf_fn: Callable[..., Any],
    include_continuous_info: bool = False,
    obs_scale_from_env_fn: Callable[[Any], tuple[Any, Any]] = obs_scale_from_env,
) -> EnvSetup:
    p_seed = resolve_problem_seed(seed=int(seed), problem_seed=problem_seed)
    n_seed = resolve_noise_seed_0(problem_seed=p_seed, noise_seed_0=noise_seed_0)

    maybe_register_atari_dm_backends(str(env_tag))

    env_conf = get_env_conf_fn(
        str(env_tag),
        problem_seed=p_seed,
        noise_seed_0=n_seed,
        from_pixels=bool(from_pixels),
        pixels_only=bool(pixels_only),
    )

    if not include_continuous_info:
        return EnvSetup(env_conf=env_conf, problem_seed=p_seed, noise_seed_0=n_seed)

    env_conf.ensure_spaces()
    action_space = getattr(env_conf, "action_space", None)
    if action_space is None or not all(hasattr(action_space, attr) for attr in ("shape", "low", "high")):
        raise ValueError("Continuous env setup requires a continuous Box action space.")
    action_shape = tuple((int(v) for v in getattr(action_space, "shape", ())))
    act_dim = int(np.prod(action_shape)) if action_shape else 1
    action_low, action_high = normalize_action_bounds(
        np.asarray(action_space.low),
        np.asarray(action_space.high),
        dim=act_dim,
    )

    # Obs scaling
    obs_lb, obs_width = obs_scale_from_env_fn(env_conf)
    if obs_lb is not None:
        obs_lb = np.asarray(obs_lb, dtype=np.float32).reshape(-1)
    if obs_width is not None:
        obs_width = np.asarray(obs_width, dtype=np.float32).reshape(-1)

    return EnvSetup(
        env_conf=env_conf,
        problem_seed=p_seed,
        noise_seed_0=n_seed,
        act_dim=act_dim,
        action_low=np.asarray(action_low, dtype=np.float32),
        action_high=np.asarray(action_high, dtype=np.float32),
        obs_lb=obs_lb,
        obs_width=obs_width,
    )
