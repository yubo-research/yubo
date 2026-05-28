from __future__ import annotations

import pytest

pytest.importorskip("enn")

from unittest.mock import sentinel

from analysis.fitting_time import (
    EnnFitTimingResult,
    benchmark_enn_fit_timing,
    enn_fit_quality_ns,
)
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver


def test_benchmark_enn_fit_timing_importable_from_package():
    assert callable(benchmark_enn_fit_timing)


def test_enn_fit_quality_ns_matches_job_fit_quality():
    from analysis.fitting_time.batch_jobs import job_fit_quality

    expected = []
    for job in job_fit_quality():
        ni = int(job.n)
        if ni not in expected:
            expected.append(ni)
    assert enn_fit_quality_ns() == tuple(expected)


@pytest.mark.parametrize(
    ("n", "k_expected", "nfs_expected"),
    [(1, 1, 1), (30, 25, 10), (100, 25, 10)],
)
def test_enn_fit_k_and_num_fit_samples_boundaries(n, k_expected, nfs_expected):
    from analysis.fitting_time.fitting_time import enn_fit_k_and_num_fit_samples

    k, nfs = enn_fit_k_and_num_fit_samples(n)
    assert k == k_expected
    assert nfs == nfs_expected


def test_benchmark_enn_fit_timing_passes_hyperparams_to_enn_fit(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit as fit_mod

    captured: dict = {}

    def ctor(*args, **kwargs):
        return sentinel.model

    def fit_enn_params_capture(model, x, y, *, k, num_fit_candidates, num_fit_samples, rng, **kwargs):
        captured.update(
            k=k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            rng=rng,
        )

    monkeypatch.setattr(fit_mod, "EpistemicNearestNeighbors", ctor, raising=False)
    monkeypatch.setattr(fit_mod, "fit_enn_params", fit_enn_params_capture, raising=False)
    monkeypatch.setattr(
        fit_mod,
        "enn_test_log_likelihood",
        lambda *_args, **_kwargs: -1.0,
        raising=False,
    )

    fit_mod.benchmark_enn_fit_timing(
        D=2,
        function_name="sphere",
        data_seed=1,
        problem_seed=1,
        n=30,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )

    assert captured["k"] == 25
    assert captured["num_fit_candidates"] == 100
    assert captured["num_fit_samples"] == 10


def test_benchmark_enn_fit_timing_small_run():
    res = benchmark_enn_fit_timing(
        D=2,
        function_name="sphere",
        data_seed=0,
        problem_seed=0,
        n=3,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    assert isinstance(res, EnnFitTimingResult)
    assert res.fit_seconds > 0.0
    assert isinstance(res.log_likelihood, float)
    assert res.n == 3
    assert res.target == "sphere"
    assert res.d == 2
    assert res.problem_seed == 0
    assert res.data_seed == 0
    assert res.index_driver is EnnIncrementalIndexDriver.FLAT


def test_benchmark_enn_fit_timing_perf_only_wraps_enn_fit(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit as fit_mod

    log: list[str] = []

    def ctor(*args, **kwargs):
        log.append("ctor")
        return sentinel.model

    def fit_enn_params_nop(model, x, y, *args, **kwargs):
        log.append("enn_fit")

    def perf():
        step = sum(1 for e in log if e == "perf")
        log.append("perf")
        return float(step)

    monkeypatch.setattr(
        fit_mod.time,
        "perf_counter",
        perf,
        raising=False,
    )
    monkeypatch.setattr(fit_mod, "EpistemicNearestNeighbors", ctor, raising=False)
    monkeypatch.setattr(fit_mod, "fit_enn_params", fit_enn_params_nop, raising=False)
    monkeypatch.setattr(
        fit_mod,
        "enn_test_log_likelihood",
        lambda *_args, **_kwargs: -1.0,
        raising=False,
    )

    res = benchmark_enn_fit_timing(
        D=2,
        function_name="sphere",
        data_seed=0,
        problem_seed=0,
        n=3,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    assert log[0] == "ctor"
    assert log[-3:] == ["perf", "enn_fit", "perf"]
    assert log.count("perf") == 2
    assert res.fit_seconds >= 0.0
    assert res.log_likelihood == -1.0


@pytest.mark.parametrize(
    "driver",
    (EnnIncrementalIndexDriver.FLAT, EnnIncrementalIndexDriver.HNSW),
)
def test_benchmark_enn_fit_flat_and_hnsw_n3(driver):
    res = benchmark_enn_fit_timing(
        D=2,
        function_name="sphere",
        data_seed=1,
        problem_seed=1,
        n=3,
        index_driver=driver,
    )
    assert res.n == 3


def test_enn_fit_timing_result_exposes_orchestrator_problem_seed():
    from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed

    base_problem_seed = 17
    data_seed = synthetic_benchmark_data_seed(
        function_name="sphere",
        problem_seed=base_problem_seed,
        rep_index=3,
    )
    res = benchmark_enn_fit_timing(
        D=2,
        function_name="sphere",
        data_seed=data_seed,
        problem_seed=base_problem_seed,
        n=3,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    assert res.problem_seed == base_problem_seed
    assert res.data_seed == data_seed


def test_enn_fit_timing_enn_fit_rng_varies_with_data_seed(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit as fit_mod

    captured: list[object] = []

    def ctor(*args, **kwargs):
        return sentinel.model

    def fit_enn_params_capture(model, x, y, *, rng, **kwargs):
        captured.append(rng)

    monkeypatch.setattr(fit_mod, "EpistemicNearestNeighbors", ctor, raising=False)
    monkeypatch.setattr(fit_mod, "fit_enn_params", fit_enn_params_capture, raising=False)
    monkeypatch.setattr(
        fit_mod,
        "enn_test_log_likelihood",
        lambda *_args, **_kwargs: -1.0,
        raising=False,
    )
    from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed

    seeds = [
        synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=0),
        synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=1),
    ]
    for ds in seeds:
        fit_mod.benchmark_enn_fit_timing(
            D=2,
            function_name="sphere",
            data_seed=ds,
            problem_seed=17,
            n=3,
            index_driver=EnnIncrementalIndexDriver.FLAT,
        )
    assert len(captured) == 2
    draws = [float(rng.random()) for rng in captured]
    assert draws[0] != draws[1]
