from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("enn")

from analysis.fitting_time import EnnFitIndTimingResult, benchmark_enn_fit_ind_timing
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver


def _fake_train_rows_at(indices):
    idx = list(indices)
    return (
        np.zeros((len(idx), 2), dtype=np.float64),
        np.zeros((len(idx), 1), dtype=np.float64),
        None,
    )


class _FakeEnnFitModel:
    def __init__(self, *, n_rows: int = 0, on_sync=None):
        self._n_rows = int(n_rows)
        self._on_sync = on_sync
        self.add_count = 0

    def __len__(self):
        return self._n_rows

    def ensure_index_sync(self):
        if self._on_sync is not None:
            self._on_sync()

    def train_rows_at(self, indices):
        return _fake_train_rows_at(indices)

    def add(self, *_args, **_kwargs):
        self.add_count += 1


def test_benchmark_enn_fit_ind_timing_importable_from_package():
    assert callable(benchmark_enn_fit_ind_timing)


def test_benchmark_enn_fit_ind_passes_hyperparams_to_enn_fit(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit_ind as fit_ind_mod

    captured: list[dict] = []

    def ctor(*_args, **_kwargs):
        return _FakeEnnFitModel()

    def fit_enn_params_capture(_model, _x, _y, *, k, num_fit_candidates, num_fit_samples, rng, **kwargs):
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
    monkeypatch.setattr("optimizer.uhd_enn_fit_helpers.fit_enn_params", fit_enn_params_capture, raising=False)
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
    assert all(c["num_fit_candidates"] == 1 for c in captured)
    assert len(captured) == 1


def test_add_segment_fits_with_probability(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit_ind as fit_ind_mod

    class _FakeRng:
        def __init__(self, values):
            self._values = iter(values)

        def random(self):
            return next(self._values)

    calls: list[int] = []

    def fake_fit(_model, *, current_n, rng, params_warm_start):
        calls.append(int(current_n))
        return f"params-{current_n}", 0.5

    monkeypatch.setattr(
        fit_ind_mod,
        "_enn_fit_timed_after_add",
        fake_fit,
        raising=False,
    )

    model = _FakeEnnFitModel()
    params, fit_total = fit_ind_mod._add_segment_with_per_point_fit(
        model,
        x_seg=np.zeros((3, 2), dtype=np.float64),
        y_seg=np.zeros((3, 1), dtype=np.float64),
        yvar_row=np.ones((1, 1), dtype=np.float64),
        start_n=10,
        rng=_FakeRng([0.95, 0.01, 0.9]),
        params_warm_start=None,
    )

    assert model.add_count == 3
    assert calls == [12]
    assert params == "params-12"
    assert fit_total == 0.5


def test_timed_fit_syncs_index_before_timer(monkeypatch):
    import analysis.fitting_time.fitting_time_enn_fit_ind as fit_ind_mod

    calls: list[object] = []
    synced = False

    def fake_fit_enn_params(_model, _x, _y, **kwargs):
        calls.append(kwargs["params_warm_start"])
        return "timed-params"

    def _mark_synced():
        nonlocal synced
        synced = True

    tick = iter([10.0, 12.5])

    monkeypatch.setattr("optimizer.uhd_enn_fit_helpers.fit_enn_params", fake_fit_enn_params, raising=False)
    monkeypatch.setattr(fit_ind_mod.time, "perf_counter", lambda: next(tick))

    params, elapsed = fit_ind_mod._enn_fit_timed_after_add(
        _FakeEnnFitModel(n_rows=1, on_sync=_mark_synced),
        current_n=30,
        rng=np.random.default_rng(0),
        params_warm_start="previous-params",
    )

    assert synced
    assert calls == ["previous-params"]
    assert params == "timed-params"
    assert elapsed == 2.5


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
