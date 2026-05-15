from __future__ import annotations

from dataclasses import dataclass

REP_INDEX_BASE = 18


@dataclass(frozen=True)
class ResolvedSeeds:
    problem_seed: int
    noise_seed_0: int


def problem_seed_from_rep_index(i_rep: int) -> int:
    return REP_INDEX_BASE + int(i_rep)


def noise_seed_0_from_problem_seed(problem_seed: int) -> int:
    return 10 * int(problem_seed)


def global_seed_for_run(problem_seed: int) -> int:
    return int(problem_seed) + 27


def resolve_problem_seed(*, seed: int, problem_seed: int | None) -> int:
    if problem_seed is not None:
        return int(problem_seed)
    return problem_seed_from_rep_index(max(0, int(seed)))


def resolve_noise_seed_0(*, problem_seed: int, noise_seed_0: int | None) -> int:
    if noise_seed_0 is not None:
        return int(noise_seed_0)
    return noise_seed_0_from_problem_seed(int(problem_seed))


def resolve_run_seeds(*, seed: int, problem_seed: int | None, noise_seed_0: int | None) -> ResolvedSeeds:
    p_seed = resolve_problem_seed(seed=int(seed), problem_seed=problem_seed)
    n_seed = resolve_noise_seed_0(problem_seed=int(p_seed), noise_seed_0=noise_seed_0)
    return ResolvedSeeds(problem_seed=int(p_seed), noise_seed_0=int(n_seed))
