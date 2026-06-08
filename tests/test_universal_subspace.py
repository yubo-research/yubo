from __future__ import annotations

import pytest


def test_universal_subspace_seed_matches_expected_formula() -> None:
    from llm.universal_subspace import universal_subspace_seed

    # base_seed + num_pop_pairs * search_dim * pop_step + pop_pair_idx * search_dim
    assert (
        universal_subspace_seed(
            base_seed=10,
            num_pop_pairs=8,
            search_dim=4,
            pop_step=3,
            pop_pair_idx=2,
        )
        == 10 + 8 * 4 * 3 + 2 * 4
    )


def test_universal_global_pop_pair_idx_covers_global_range() -> None:
    from llm.universal_subspace import universal_global_pop_pair_idx

    population_size = 16
    num_engines = 4
    loras_per_engine = population_size // num_engines
    assert loras_per_engine == 4
    num_pairs_per_engine = loras_per_engine // 2
    assert num_pairs_per_engine == 2

    seen = []
    for engine_rank in range(num_engines):
        for local_pair_idx in range(num_pairs_per_engine):
            seen.append(
                universal_global_pop_pair_idx(
                    engine_rank=engine_rank,
                    num_engines=num_engines,
                    population_size=population_size,
                    local_pop_pair_idx=local_pair_idx,
                )
            )

    assert seen == list(range(population_size // 2))


def test_validate_eggroll_population_requires_even_arms_per_engine() -> None:
    from llm.es import validate_eggroll_population

    with pytest.raises(ValueError, match="arms/engine"):
        validate_eggroll_population(
            population_size=6,
            num_engines=2,  # 3 arms/engine (odd) breaks antithetic pairing per engine
            samples_per_prompt=1,
            temperature=0.0,
            pass_at_k=False,
        )
