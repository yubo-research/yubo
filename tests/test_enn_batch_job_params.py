from __future__ import annotations

from pathlib import Path

import pytest

from analysis.fitting_time.fitting_time_enn_incremental import (
    ENN_INCREMENTAL_CHECKPOINT_NS,
    enn_incremental_checkpoint_ns,
)
from experiments.enn_batch_job_params import (
    ENN_BATCH_BENCHMARK_FUNCTIONS,
    enn_batch_checkpoint_ns,
    enn_batch_shared_params,
)


def test_enn_batch_checkpoint_ns_matches_incremental_source():
    assert enn_batch_checkpoint_ns() == enn_incremental_checkpoint_ns()
    assert enn_batch_checkpoint_ns() == ENN_INCREMENTAL_CHECKPOINT_NS


def test_enn_batch_shared_params_uses_shared_grid():
    shared = enn_batch_shared_params(num_reps=10, d=2, problem_seed=17)
    assert shared.benchmark_functions == ENN_BATCH_BENCHMARK_FUNCTIONS
    assert shared.checkpoint_ns == enn_batch_checkpoint_ns()
    assert shared.d == 2
    assert shared.problem_seed == 17
    assert shared.num_reps == 10


def test_iter_fit_jobs_uses_same_checkpoint_ns_as_shared_params(tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    jobs = list(impl._iter_fit_jobs(tmp_path, "flat", 1, 2, 17))
    fit_ns = sorted({job[1][2] for job in jobs})
    shared = enn_batch_shared_params(num_reps=1, d=2, problem_seed=17)
    assert fit_ns == sorted(shared.checkpoint_ns)


def test_validate_enn_batch_scalars_rejects_bad_inputs():
    with pytest.raises(ValueError, match="num_reps"):
        enn_batch_shared_params(num_reps=0, d=2, problem_seed=17)
    with pytest.raises(ValueError, match="D must be positive"):
        enn_batch_shared_params(num_reps=1, d=0, problem_seed=17)
