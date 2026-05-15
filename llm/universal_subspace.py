from __future__ import annotations


def universal_subspace_seed(
    *,
    base_seed: int,
    num_pop_pairs: int,
    search_dim: int,
    pop_step: int,
    pop_pair_idx: int,
) -> int:
    """Seed schedule shared by universal-subspace perturb/apply paths.

    Keep this in lock-step across:
    - evaluation-time perturbations (per arm)
    - ES update reconstruction (per population pair)
    """

    return int(base_seed) + int(num_pop_pairs) * int(search_dim) * int(pop_step) + int(pop_pair_idx) * int(search_dim)


def universal_global_pop_pair_idx(
    *,
    engine_rank: int,
    num_engines: int,
    population_size: int,
    local_pop_pair_idx: int,
) -> int:
    """Maps an engine-local pop-pair index to the global pop-pair index."""

    if int(num_engines) < 1:
        raise ValueError("num_engines must be >= 1.")
    if int(population_size) % int(num_engines) != 0:
        raise ValueError("population_size must be divisible by num_engines.")
    loras_per_engine = int(population_size) // int(num_engines)
    if int(loras_per_engine) % 2 != 0:
        raise ValueError("population_size/num_engines must be even for antithetic sampling.")
    num_pairs_per_engine = int(loras_per_engine) // 2
    return int(engine_rank) * int(num_pairs_per_engine) + int(local_pop_pair_idx)


__all__ = ["universal_global_pop_pair_idx", "universal_subspace_seed"]
