from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from common.experiment_seeds import global_seed_for_run, noise_seed_0_from_problem_seed, problem_seed_from_rep_index


def resolve_problem_seed(*, seed: int, problem_seed: int | None) -> int:
    return int(problem_seed) if problem_seed is not None else problem_seed_from_rep_index(max(0, int(seed)))


def resolve_noise_seed_0(*, problem_seed: int, noise_seed_0: int | None) -> int:
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


def resolve_run_seeds(*, seed: int, problem_seed: int | None, noise_seed_0: int | None) -> ResolvedSeeds:
    resolved_problem_seed = resolve_problem_seed(seed=int(seed), problem_seed=problem_seed)
    resolved_noise_seed_0 = resolve_noise_seed_0(problem_seed=int(resolved_problem_seed), noise_seed_0=noise_seed_0)
    return ResolvedSeeds(problem_seed=int(resolved_problem_seed), noise_seed_0=int(resolved_noise_seed_0))


def build_seeded_env_conf(
    *, env_tag: str, problem_seed: int, noise_seed_0: int, from_pixels: bool, pixels_only: bool, get_env_conf_fn: Callable[..., Any]
) -> SeededEnvConf:
    env_conf = get_env_conf_fn(
        str(env_tag), problem_seed=int(problem_seed), noise_seed_0=int(noise_seed_0), from_pixels=bool(from_pixels), pixels_only=bool(pixels_only)
    )
    return SeededEnvConf(env_conf=env_conf, problem_seed=int(problem_seed), noise_seed_0=int(noise_seed_0))


def build_seeded_env_conf_from_run(
    *, env_tag: str, seed: int, problem_seed: int | None, noise_seed_0: int | None, from_pixels: bool, pixels_only: bool, get_env_conf_fn: Callable[..., Any]
) -> SeededEnvConf:
    resolved = resolve_run_seeds(seed=int(seed), problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    return build_seeded_env_conf(
        env_tag=str(env_tag),
        problem_seed=int(resolved.problem_seed),
        noise_seed_0=int(resolved.noise_seed_0),
        from_pixels=bool(from_pixels),
        pixels_only=bool(pixels_only),
        get_env_conf_fn=get_env_conf_fn,
    )


__all__ = [
    "ResolvedSeeds",
    "SeededEnvConf",
    "build_seeded_env_conf",
    "build_seeded_env_conf_from_run",
    "global_seed_for_run",
    "resolve_noise_seed_0",
    "resolve_problem_seed",
    "resolve_run_seeds",
]
