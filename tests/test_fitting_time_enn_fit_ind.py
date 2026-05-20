from __future__ import annotations

import pytest

pytest.importorskip("enn")

from analysis.fitting_time import EnnFitIndTimingResult, benchmark_enn_fit_ind_timing
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver


def test_benchmark_enn_fit_ind_timing_importable_from_package():
    assert callable(benchmark_enn_fit_ind_timing)


def test_benchmark_enn_fit_ind_passes_hyperparams_to_enn_fit(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit_ind as fit_ind_mod

    captured: list[dict] = []

    class _FakeModel:
        def add(self, *_args, **_kwargs):
            pass

    def ctor(*_args, **_kwargs):
        return _FakeModel()

    def enn_fit_capture(_model, *, k, num_fit_candidates, num_fit_samples, rng, **kwargs):
        captured.append(
            {
                "k": k,
                "num_fit_candidates": num_fit_candidates,
                "num_fit_samples": num_fit_samples,
                "params_warm_start": kwargs.get("params_warm_start"),
            }
        )
        from enn.enn.enn_params import ENNParams

        return ENNParams(
            k_num_neighbors=int(k),
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        )

    monkeypatch.setattr(
        "enn.enn.enn_class.EpistemicNearestNeighbors",
        ctor,
        raising=False,
    )
    monkeypatch.setattr("enn.enn.enn_fit.enn_fit", enn_fit_capture, raising=False)
    monkeypatch.setattr(
        fit_ind_mod,
        "enn_test_log_likelihood",
        lambda *_args, **_kwargs: -1.0,
        raising=False,
    )

    res = fit_ind_mod.benchmark_enn_fit_ind_timing(
        D=2,
        function_name="sphere",
        problem_seed=1,
        index_driver=EnnIncrementalIndexDriver.FLAT,
        checkpoints=(1, 3),
    )

    assert isinstance(res, EnnFitIndTimingResult)
    assert res.n == (1, 3)
    assert len(res.fit_seconds) == 2
    assert len(res.log_likelihood) == 2
    assert captured[0]["num_fit_candidates"] == 1
    assert captured[0]["params_warm_start"] is None
    assert captured[1]["params_warm_start"] is not None
    assert all(c["num_fit_candidates"] == 1 for c in captured)


def test_benchmark_enn_fit_ind_small_run():
    res = benchmark_enn_fit_ind_timing(
        D=2,
        function_name="sphere",
        problem_seed=0,
        index_driver=EnnIncrementalIndexDriver.FLAT,
        checkpoints=(1, 3),
    )
    assert res.n == (1, 3)
    assert all(t >= 0.0 for t in res.fit_seconds)
    assert len(res.log_likelihood) == 2
