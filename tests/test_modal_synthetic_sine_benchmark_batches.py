"""Modal disk run, jobs loader, and batch worker tests for synthetic sine benchmark."""

from __future__ import annotations

import contextlib
import types
from collections import namedtuple
from pathlib import Path

import pytest

from analysis.fitting_time.batch_jobs import (
    SyntheticBenchJob,
    example_two_targets_n12_d2,
)
from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    synthetic_benchmark_data_seed,
)
from experiments.synthetic_sine_benchmark_payload import (
    META_KEY,
    build_synthetic_sine_benchmark_remote_payload,
    load_synthetic_sine_benchmark_jobs,
    read_synthetic_sine_benchmark_json,
    run_synthetic_sine_benchmark_modal_to_disk,
    synthetic_sine_benchmark_from_payload,
    synthetic_sine_benchmark_result_to_payload,
    write_synthetic_sine_benchmark_json,
)
from tests.synthetic_sine_benchmark_helpers import bench_result, make_surrogate_benchmark


def test_payload_run_synthetic_sine_benchmark_modal_to_disk(monkeypatch, tmp_path: Path):
    import experiments.synthetic_sine_benchmark_payload as pl

    z = bench_result(0.0, 0.0, 0.0)
    zero = make_surrogate_benchmark(enn=z, smac_rf=z, dngo=z, exact_gp=z, svgp_default=z, svgp_linear=z, vecchia=z)

    _App = type("_App", (), {"run": lambda self: contextlib.nullcontext()})

    def _remote(n, d, fn, ps, *_args):
        return pl.synthetic_sine_benchmark_result_to_payload(zero, n=n, d=d, function_name=fn, problem_seed=ps)

    _Rem = type("_Rem", (), {"remote": staticmethod(_remote)})

    monkeypatch.setattr(pl.modal, "enable_output", lambda: contextlib.nullcontext())
    dest = run_synthetic_sine_benchmark_modal_to_disk(2, 2, "sphere", 3, tmp_path, app=_App(), remote_fn=_Rem())
    assert dest.exists() and "sphere" in dest.name
    with pytest.raises(ValueError, match="non-empty"):
        run_synthetic_sine_benchmark_modal_to_disk(2, 2, "  \t", 4, tmp_path / "ws", app=_App(), remote_fn=_Rem())


def test_load_synthetic_sine_benchmark_jobs_real_module():
    rows = load_synthetic_sine_benchmark_jobs("example_two_targets_n12_d2")
    assert rows == [(12, 2, "sphere", 0), (12, 2, "sine", 0)]
    assert example_two_targets_n12_d2() == [
        SyntheticBenchJob(n=12, d=2, target="sphere", problem_seed=0),
        SyntheticBenchJob(n=12, d=2, target="sine", problem_seed=0),
    ]


def test_load_example_sphere_n12_d2_seed0():
    assert load_synthetic_sine_benchmark_jobs("example_sphere_n12_d2_seed0") == [(12, 2, "sphere", 0)]


@pytest.mark.parametrize(
    ("jobs_fn", "match"),
    [
        ("12bad", "identifier"),
        ("missing_fn_xyz", "missing or not callable"),
    ],
)
def test_load_synthetic_sine_benchmark_jobs_errors(jobs_fn, match):
    with pytest.raises(ValueError, match=match):
        load_synthetic_sine_benchmark_jobs(jobs_fn)


def test_load_synthetic_sine_benchmark_jobs_injected_module():
    _J = namedtuple("SyntheticBenchJob", ["n", "d", "target", "problem_seed"], defaults=(0,))

    def _fn():
        return [_J(2, 2, "sphere", 0)]

    def _empty():
        return []

    def _bad_elem():
        return [object()]

    m = types.SimpleNamespace(SyntheticBenchJob=_J, good=_fn, empty=_empty, bad_elem=_bad_elem)
    assert load_synthetic_sine_benchmark_jobs("good", _batch_jobs_module=m) == [(2, 2, "sphere", 0)]
    with pytest.raises(ValueError, match="non-empty"):
        load_synthetic_sine_benchmark_jobs("empty", _batch_jobs_module=m)
    with pytest.raises(TypeError, match="SyntheticBenchJob"):
        load_synthetic_sine_benchmark_jobs("bad_elem", _batch_jobs_module=m)


def test_batches_impl_iter_missing_jobs_respects_existing_files(tmp_path: Path):
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = impl._benchmark_json_dest(
        out_dir,
        n=12,
        d=2,
        function_name="sphere",
        problem_seed=0,
        num_reps=4,
    )
    existing.write_text("{}")

    jobs = list(impl._iter_missing_jobs("example_two_targets_n12_d2", out_dir, 4))

    assert len(jobs) == 4
    key, payload = jobs[0]
    assert key == "N12_D2_sine_pseed0_nrep4-rep0"
    assert payload == (12, 2, "sine", 0, 0, 4)


def test_batches_impl_aggregate_reps_to_dest(tmp_path: Path):
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    out_dir = tmp_path / "out"
    rep0 = make_surrogate_benchmark(
        enn=bench_result(1.0, 2.0, 3.0),
        smac_rf=bench_result(2.0, 3.0, 4.0),
        dngo=bench_result(3.0, 4.0, 5.0),
        exact_gp=bench_result(4.0, 5.0, 6.0),
        svgp_default=bench_result(5.0, 6.0, 7.0),
        svgp_linear=bench_result(6.0, 7.0, 8.0),
        vecchia=bench_result(7.0, 8.0, 9.0),
    )
    rep1 = make_surrogate_benchmark(
        enn=bench_result(3.0, 4.0, 5.0),
        smac_rf=bench_result(4.0, 5.0, 6.0),
        dngo=bench_result(5.0, 6.0, 7.0),
        exact_gp=bench_result(6.0, 7.0, 8.0),
        svgp_default=bench_result(7.0, 8.0, 9.0),
        svgp_linear=bench_result(8.0, 9.0, 10.0),
        vecchia=bench_result(9.0, 10.0, 11.0),
    )
    rep0_path = impl._rep_json_dest(out_dir, n=10, d=3, function_name="sphere", problem_seed=7, rep_index=0)
    rep1_path = impl._rep_json_dest(out_dir, n=10, d=3, function_name="sphere", problem_seed=7, rep_index=1)
    write_synthetic_sine_benchmark_json(
        rep0_path,
        synthetic_sine_benchmark_result_to_payload(rep0, n=10, d=3, function_name="sphere", problem_seed=7, num_reps=1),
    )
    write_synthetic_sine_benchmark_json(
        rep1_path,
        synthetic_sine_benchmark_result_to_payload(rep1, n=10, d=3, function_name="sphere", problem_seed=7, num_reps=1),
    )
    _, rep1_meta = read_synthetic_sine_benchmark_json(rep1_path)
    assert rep1_meta["problem_seed"] == 7

    dest = impl._aggregate_reps_to_dest(out_dir, n=10, d=3, function_name="sphere", problem_seed=7, num_reps=2)

    assert dest == impl._benchmark_json_dest(out_dir, n=10, d=3, function_name="sphere", problem_seed=7, num_reps=2)
    bench, meta = read_synthetic_sine_benchmark_json(dest)
    assert meta == {
        "N": 10,
        "D": 3,
        "function_name": "sphere",
        "problem_seed": 7,
        "num_reps": 2,
    }
    assert bench.results["enn"].fit_seconds.mu == 2.0
    assert bench.results["enn"].normalized_rmse.mu == 3.0
    assert bench.results["enn"].log_likelihood.mu == 4.0


def test_batches_impl_worker_uses_function_and_rep_specific_data_seed(monkeypatch):
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    store = {}
    captured = {}

    def fake_single(*, N, D, function_name, surrogate_key, data_seed):
        captured["args"] = (N, D, function_name, surrogate_key, data_seed)
        return (0.1, 0.2, 0.3)

    monkeypatch.setattr(impl, "benchmark_single_surrogate_with_data", fake_single)
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)

    impl.synthetic_sine_benchmark_batch_worker.info.raw_f(("tag-x", 12, 2, "sphere", 17, 3, 10, "enn"))

    expected_data_seed = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=3)
    assert captured["args"] == (12, 2, "sphere", "enn", expected_data_seed)
    jk = impl._job_key(
        n=12,
        d=2,
        function_name="sphere",
        problem_seed=17,
        num_reps=10,
        surrogate_key="enn",
    )
    payload, *_rest = store[f"{jk}-rep3"]
    assert payload[META_KEY]["problem_seed"] == 17
    assert payload[META_KEY]["data_seed"] == expected_data_seed
    assert payload[META_KEY]["rep_index"] == 3
    assert payload[META_KEY]["surrogate_key"] == "enn"


def test_batches_impl_submit_missing_prints_data_seed_ranges(monkeypatch, capsys):
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    spawned = []

    def _spawn(_self, batch, tag):
        spawned.append((list(batch), tag))

    _SpawnFn = type("_SpawnFn", (), {"spawn": _spawn})

    monkeypatch.setattr(
        impl.modal.Function,
        "from_name",
        lambda *_args, **_kwargs: _SpawnFn(),
    )
    monkeypatch.setattr(
        impl,
        "_iter_missing_surrogate_jobs",
        lambda *_args, **_kwargs: iter(
            [
                ("k0", (12, 2, "sphere", 17, 0, 4, "enn")),
                ("k1", (12, 2, "sphere", 17, 0, 4, "smac_rf")),
                ("k2", (12, 2, "sphere", 17, 2, 4, "enn")),
                ("k3", (12, 2, "sphere", 17, 3, 4, "dngo")),
            ]
        ),
    )

    impl._submit_missing("tag-x", "jobs_fn", "out_dir", 4)

    out = capsys.readouterr().out
    seed0 = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=0)
    seed3 = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=3)
    assert spawned == [
        (
            [
                ("k0", (12, 2, "sphere", 17, 0, 4, "enn")),
                ("k1", (12, 2, "sphere", 17, 0, 4, "smac_rf")),
                ("k2", (12, 2, "sphere", 17, 2, 4, "enn")),
                ("k3", (12, 2, "sphere", 17, 3, 4, "dngo")),
            ],
            "tag-x",
        )
    ]
    assert f"data_seed range N=12 D=2 fn=sphere pseed=17 reps=0-3 (3/4) seeds={seed0}-{seed3}" in out
    assert "surrogate_jobs=4" in out
    assert "submitted 4 jobs" in out


def test_build_synthetic_sine_benchmark_remote_payload_delegates(monkeypatch):
    """Heavy ``benchmark_synthetic_sine_surrogates`` smoke lives in ``test_evaluate``; mock here."""
    captured: dict = {}

    def _fake_benchmark(*, N, D, function_name, problem_seed, num_reps=1, b_fast_only=False):
        captured["N"] = N
        captured["D"] = D
        captured["function_name"] = function_name
        captured["problem_seed"] = problem_seed
        captured["num_reps"] = num_reps
        captured["b_fast_only"] = b_fast_only
        nan = float("nan")
        return make_surrogate_benchmark(
            enn=bench_result(0.1, 0.2, -1.0),
            smac_rf=bench_result(nan, nan, nan),
            dngo=bench_result(0.3, 0.4, -2.0),
            exact_gp=bench_result(0.5, 0.6, -3.0),
            svgp_default=bench_result(0.7, 0.8, -4.0),
            svgp_linear=bench_result(0.9, 1.0, -5.0),
            vecchia=bench_result(nan, nan, nan),
        )

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.benchmark_synthetic_sine_surrogates",
        _fake_benchmark,
    )
    payload = build_synthetic_sine_benchmark_remote_payload(11, 3, "ackley", 42)
    bench, meta = synthetic_sine_benchmark_from_payload(payload)
    assert captured == {
        "N": 11,
        "D": 3,
        "function_name": "ackley",
        "problem_seed": 42,
        "num_reps": 1,
        "b_fast_only": False,
    }
    assert bench.results["enn"].fit_seconds.mu == 0.1
    assert meta["function_name"] == "ackley"


def test_iter_missing_surrogate_jobs_ignores_aggregate_json(tmp_path: Path):
    from experiments.modal_synthetic_sine_benchmark_batches_reps import (
        benchmark_json_dest,
        iter_missing_surrogate_jobs,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    agg = benchmark_json_dest(
        out_dir,
        n=12,
        d=2,
        function_name="sphere",
        problem_seed=0,
        num_reps=4,
    )
    agg.write_text("{}")

    jobs = list(iter_missing_surrogate_jobs("example_two_targets_n12_d2", out_dir, 4))
    assert len(jobs) == 4 * len(SURROGATE_BENCHMARK_KEYS) * 2


def test_batches_impl_collect_writes_only_surrogate_shard(tmp_path: Path, monkeypatch):
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    surr_payload = {
        "triple": [1.0, 2.0, 3.0],
        META_KEY: {
            "N": 10,
            "D": 3,
            "function_name": "sphere",
            "problem_seed": 7,
            "data_seed": 0,
            "rep_index": 0,
            "surrogate_key": "enn",
        },
    }
    store = {"k0": (surr_payload, 10, 3, "sphere", 7, 0, 2, "enn")}
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)
    monkeypatch.setattr(
        impl.modal.Function,
        "from_name",
        lambda *_a, **_k: type("_F", (), {"spawn": staticmethod(lambda *_x, **_y: None)})(),
    )
    impl._collect("tag-x", tmp_path)
    dest = impl._surrogate_rep_json_dest(
        tmp_path,
        n=10,
        d=3,
        function_name="sphere",
        problem_seed=7,
        rep_index=0,
        surrogate_key="enn",
    )
    assert dest.exists()
    rep_path = impl._rep_json_dest(tmp_path, n=10, d=3, function_name="sphere", problem_seed=7, rep_index=0)
    assert not rep_path.exists()
    top = impl._benchmark_json_dest(tmp_path, n=10, d=3, function_name="sphere", problem_seed=7, num_reps=2)
    assert not top.exists()


def test_batches_impl_worker_rejects_seven_tuple_job():
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    with pytest.raises(ValueError, match="expected 8-tuple"):
        impl.synthetic_sine_benchmark_batch_worker.info.raw_f(("tag-x", 12, 2, "sphere", 17, 3, 10))
