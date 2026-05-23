from __future__ import annotations

import pytest

from analysis.fitting_time import (
    EnnFullOptTimingResult,
    benchmark_enn_full_optimization_proposal_timing,
)
from analysis.fitting_time.fitting_time_enn_full_opt import (
    _finalize_stop_reason,
    _snapshots_from_iter_counts,
    _wall_clock_stop_requested,
    collect_full_opt_snapshots_from_optimizer,
    opt_name_for_index_driver,
)
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver


def test_benchmark_enn_full_optimization_importable_from_package():
    from analysis.fitting_time import benchmark_enn_full_optimization_proposal_timing as exported

    assert exported is benchmark_enn_full_optimization_proposal_timing


def test_opt_name_for_index_driver():
    assert opt_name_for_index_driver(EnnIncrementalIndexDriver.FLAT) == "turbo-enn-fit-ucb"
    assert opt_name_for_index_driver(EnnIncrementalIndexDriver.HNSW) == "turbo-enn-fit-ucb/idx=hnsw"


def test_benchmark_rejects_problem_seed_inconsistent_with_rep_index():
    from common.experiment_seeds import problem_seed_from_rep_index

    with pytest.raises(ValueError, match="problem_seed.*rep_index"):
        benchmark_enn_full_optimization_proposal_timing(
            env_tag="f:ackley-3d",
            problem_seed=problem_seed_from_rep_index(0) + 1,
            rep_index=0,
            checkpoints=(1,),
            num_rounds=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_arms": 2}, "num_arms"),
        ({"num_denoise": 2}, "num_denoise"),
        ({"policy_tag": "linear"}, "policy_tag"),
    ],
)
def test_benchmark_rejects_non_plan_q5_hyperparameters(kwargs, match):
    """Plan Q5: num_arms=1, num_denoise=1, policy_tag=pure-function are fixed."""
    with pytest.raises(ValueError, match=match):
        benchmark_enn_full_optimization_proposal_timing(
            env_tag="f:ackley-3d",
            problem_seed=18,
            rep_index=0,
            checkpoints=(1,),
            num_rounds=1,
            **kwargs,
        )


def test_finalize_stop_reason_wall_clock_limit():
    _wall_clock_stop_requested[0] = True
    try:
        reason = _finalize_stop_reason(
            next_idx=1,
            num_checkpoints=3,
            i_iter=5,
            num_rounds=100,
        )
        assert reason == "wall_clock_limit"
    finally:
        _wall_clock_stop_requested[0] = False


def test_benchmark_stop_reason_num_rounds_when_checkpoints_incomplete():
    result = benchmark_enn_full_optimization_proposal_timing(
        env_tag="f:ackley-3d",
        problem_seed=18,
        rep_index=0,
        checkpoints=(1, 3, 10),
        num_rounds=2,
    )
    assert result.n == (1,)
    assert result.stop_reason == "num_rounds"


def test_snapshots_from_iter_counts():
    ns: list[int] = []
    elapsed: list[float] = []
    idx = _snapshots_from_iter_counts(1, 0.5, (1, 3, 10), 0, ns, elapsed)
    assert ns == [1]
    assert elapsed == [0.5]
    idx = _snapshots_from_iter_counts(3, 1.2, (1, 3, 10), idx, ns, elapsed)
    assert ns == [1, 3]
    assert elapsed == [0.5, 1.2]
    idx = _snapshots_from_iter_counts(2, 0.9, (1, 3, 10), idx, ns, elapsed)
    assert idx == 2
    assert ns == [1, 3]


def test_collect_full_opt_snapshots_from_optimizer():
    class _FakeOpt:
        def __init__(self):
            self._i_iter = 0
            self._cum_dt_proposing = 0.0
            self.stopped = False

        def iterate(self):
            self._i_iter += 1
            self._cum_dt_proposing += float(self._i_iter) * 0.1

        def stop(self):
            self.stopped = True

    opt = _FakeOpt()
    ns, elapsed, stop_reason = collect_full_opt_snapshots_from_optimizer(
        opt,
        checkpoints=(1, 3),
        max_iterations=5,
    )
    assert ns == (1, 3)
    assert elapsed == pytest.approx((0.1, 0.6))
    assert stop_reason == "completed"
    assert opt.stopped


@pytest.mark.parametrize(
    "index_driver",
    [EnnIncrementalIndexDriver.FLAT, EnnIncrementalIndexDriver.HNSW],
)
def test_benchmark_enn_full_opt_ackley_3d_smoke(index_driver):
    result = benchmark_enn_full_optimization_proposal_timing(
        env_tag="f:ackley-3d",
        problem_seed=18,
        rep_index=0,
        index_driver=index_driver,
        checkpoints=(1, 3),
        num_rounds=10,
    )
    assert isinstance(result, EnnFullOptTimingResult)
    assert result.n == (1, 3)
    assert len(result.proposal_elapsed_seconds) == 2
    assert result.env_tag == "f:ackley-3d"
    assert result.opt_name == opt_name_for_index_driver(index_driver)
    assert all(t >= 0.0 for t in result.proposal_elapsed_seconds)
