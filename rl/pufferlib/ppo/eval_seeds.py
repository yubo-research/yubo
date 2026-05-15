from __future__ import annotations

from common.experiment_seeds import resolve_run_seeds


def resolve_eval_seeds(config) -> tuple[int, int]:
    resolved = resolve_run_seeds(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    return (int(resolved.problem_seed), int(resolved.noise_seed_0))
