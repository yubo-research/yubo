from common.experiment_seeds import (
    noise_seed_0_from_problem_seed,
    problem_seed_from_rep_index,
)

# global_seed_for_run is imported lazily inside the function


def resolve_problem_seed(*, seed: int, problem_seed: int | None) -> int:
    return int(problem_seed) if problem_seed is not None else problem_seed_from_rep_index(max(0, int(seed)))


def resolve_noise_seed_0(*, problem_seed: int, noise_seed_0: int | None) -> int:
    return int(noise_seed_0) if noise_seed_0 is not None else noise_seed_0_from_problem_seed(problem_seed)


def global_seed_for_run(problem_seed: int) -> int:
    from common.experiment_seeds import global_seed_for_run as _fn

    return _fn(problem_seed)
