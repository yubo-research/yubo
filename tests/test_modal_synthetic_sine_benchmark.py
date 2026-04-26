import contextlib
import math
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from analysis.fitting_time.batch_jobs import (
    SyntheticBenchJob,
    example_two_targets_n12_d2,
)
from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    BMResult,
    MuSe,
    SyntheticSineSurrogateBenchmark,
    synthetic_benchmark_data_seed,
)
from experiments.synthetic_sine_benchmark_payload import (
    META_KEY,
    build_synthetic_sine_benchmark_remote_payload,
    load_synthetic_sine_benchmark_jobs,
    load_synthetic_sine_benchmark_json_dir,
    load_synthetic_sine_benchmark_json_dir_long,
    read_synthetic_sine_benchmark_json,
    run_synthetic_sine_benchmark_modal_to_disk,
    synthetic_sine_benchmark_config_slug,
    synthetic_sine_benchmark_from_payload,
    synthetic_sine_benchmark_rep_slug,
    synthetic_sine_benchmark_result_to_payload,
    synthetic_surrogate_benchmark_row_caption,
    wide_surrogate_benchmark_row_to_comparison_records,
    wide_surrogate_benchmark_row_to_long_records,
    write_synthetic_sine_benchmark_json,
)


def _br(
    fit_s: float,
    nrmse: float,
    ll: float,
    *,
    se_f: float = 0.0,
    se_n: float = 0.0,
    se_l: float = 0.0,
) -> BMResult:
    return BMResult(MuSe(fit_s, se_f), MuSe(nrmse, se_n), MuSe(ll, se_l))


def _bench(**kwargs: BMResult) -> SyntheticSineSurrogateBenchmark:
    return SyntheticSineSurrogateBenchmark(results={k: kwargs[k] for k in SURROGATE_BENCHMARK_KEYS})


def test_synthetic_sine_benchmark_payload_round_trip():
    nan = float("nan")
    r = _bench(
        enn=_br(0.1, 0.2, -1.0),
        smac_rf=_br(nan, nan, nan),
        dngo=_br(0.3, 0.4, -2.0),
        exact_gp=_br(0.5, 0.6, -3.0),
        svgp_default=_br(0.7, 0.8, -4.0),
        svgp_linear=_br(0.9, 1.0, -5.0),
        vecchia=_br(1.1, 1.2, -6.0),
    )
    payload = synthetic_sine_benchmark_result_to_payload(r, n=10, d=3, function_name="sphere", problem_seed=7)
    r2, meta = synthetic_sine_benchmark_from_payload(payload)
    assert meta == {
        "N": 10,
        "D": 3,
        "function_name": "sphere",
        "problem_seed": 7,
        "num_reps": 1,
    }
    assert r2 == r
    assert math.isnan(r2.results["smac_rf"].fit_seconds.mu)


def test_wide_surrogate_benchmark_row_to_comparison_records_and_caption():
    row = {
        "file": "x.json",
        "N": 1000,
        "D": 30,
        "function_name": "ackley",
        "problem_seed": 17,
        "enn_fit_seconds_mu": 0.1,
        "enn_fit_seconds_se": 0.0,
        "enn_normalized_rmse_mu": 0.2,
        "enn_normalized_rmse_se": 0.0,
        "enn_log_likelihood_mu": -1.0,
        "enn_log_likelihood_se": 0.0,
        "smac_rf_fit_seconds_mu": float("nan"),
        "smac_rf_fit_seconds_se": float("nan"),
        "smac_rf_normalized_rmse_mu": 0.0,
        "smac_rf_normalized_rmse_se": 0.0,
        "smac_rf_log_likelihood_mu": 0.0,
        "smac_rf_log_likelihood_se": 0.0,
        "dngo_fit_seconds_mu": 1.0,
        "dngo_fit_seconds_se": 0.0,
        "dngo_normalized_rmse_mu": 0.3,
        "dngo_normalized_rmse_se": 0.0,
        "dngo_log_likelihood_mu": -2.0,
        "dngo_log_likelihood_se": 0.0,
        "exact_gp_fit_seconds_mu": 1.1,
        "exact_gp_fit_seconds_se": 0.0,
        "exact_gp_normalized_rmse_mu": 0.4,
        "exact_gp_normalized_rmse_se": 0.0,
        "exact_gp_log_likelihood_mu": -3.0,
        "exact_gp_log_likelihood_se": 0.0,
        "svgp_default_fit_seconds_mu": 1.2,
        "svgp_default_fit_seconds_se": 0.0,
        "svgp_default_normalized_rmse_mu": 0.5,
        "svgp_default_normalized_rmse_se": 0.0,
        "svgp_default_log_likelihood_mu": -4.0,
        "svgp_default_log_likelihood_se": 0.0,
        "svgp_linear_fit_seconds_mu": 1.3,
        "svgp_linear_fit_seconds_se": 0.0,
        "svgp_linear_normalized_rmse_mu": 0.6,
        "svgp_linear_normalized_rmse_se": 0.0,
        "svgp_linear_log_likelihood_mu": -5.0,
        "svgp_linear_log_likelihood_se": 0.0,
        "vecchia_fit_seconds_mu": 1.4,
        "vecchia_fit_seconds_se": 0.0,
        "vecchia_normalized_rmse_mu": 0.7,
        "vecchia_normalized_rmse_se": 0.0,
        "vecchia_log_likelihood_mu": -6.0,
        "vecchia_log_likelihood_se": 0.0,
    }
    recs = wide_surrogate_benchmark_row_to_comparison_records(row)
    assert len(recs) == 7
    assert recs[0]["Surrogate"] == "ENN" and recs[0]["Fit (s) μ"] == 0.1
    assert math.isnan(recs[1]["Fit (s) μ"])
    cap = synthetic_surrogate_benchmark_row_caption(row)
    assert "N=1000" in cap and "D=30" in cap and "ackley" in cap and "x.json" in cap


def test_wide_surrogate_benchmark_row_to_long_records():
    row = {
        "file": "x.json",
        "N": 1000,
        "D": 30,
        "function_name": "ackley",
        "problem_seed": 17,
        "num_reps": 1,
        "enn_fit_seconds_mu": 0.1,
        "enn_fit_seconds_se": 0.01,
        "enn_normalized_rmse_mu": 0.2,
        "enn_normalized_rmse_se": 0.02,
        "enn_log_likelihood_mu": -1.0,
        "enn_log_likelihood_se": 0.03,
        "smac_rf_fit_seconds_mu": 0.4,
        "smac_rf_fit_seconds_se": 0.04,
        "smac_rf_normalized_rmse_mu": 0.5,
        "smac_rf_normalized_rmse_se": 0.05,
        "smac_rf_log_likelihood_mu": -2.0,
        "smac_rf_log_likelihood_se": 0.06,
        "dngo_fit_seconds_mu": 0.0,
        "dngo_fit_seconds_se": 0.0,
        "dngo_normalized_rmse_mu": 0.0,
        "dngo_normalized_rmse_se": 0.0,
        "dngo_log_likelihood_mu": 0.0,
        "dngo_log_likelihood_se": 0.0,
        "exact_gp_fit_seconds_mu": 0.0,
        "exact_gp_fit_seconds_se": 0.0,
        "exact_gp_normalized_rmse_mu": 0.0,
        "exact_gp_normalized_rmse_se": 0.0,
        "exact_gp_log_likelihood_mu": 0.0,
        "exact_gp_log_likelihood_se": 0.0,
        "svgp_default_fit_seconds_mu": 0.0,
        "svgp_default_fit_seconds_se": 0.0,
        "svgp_default_normalized_rmse_mu": 0.0,
        "svgp_default_normalized_rmse_se": 0.0,
        "svgp_default_log_likelihood_mu": 0.0,
        "svgp_default_log_likelihood_se": 0.0,
        "svgp_linear_fit_seconds_mu": 0.0,
        "svgp_linear_fit_seconds_se": 0.0,
        "svgp_linear_normalized_rmse_mu": 0.0,
        "svgp_linear_normalized_rmse_se": 0.0,
        "svgp_linear_log_likelihood_mu": 0.0,
        "svgp_linear_log_likelihood_se": 0.0,
        "vecchia_fit_seconds_mu": 0.7,
        "vecchia_fit_seconds_se": 0.07,
        "vecchia_normalized_rmse_mu": 0.8,
        "vecchia_normalized_rmse_se": 0.08,
        "vecchia_log_likelihood_mu": -3.0,
        "vecchia_log_likelihood_se": 0.09,
    }
    recs = wide_surrogate_benchmark_row_to_long_records(row)
    assert len(recs) == 7
    assert recs[0]["surrogate"] == "enn"
    assert recs[0]["fit_seconds_mu"] == 0.1
    assert recs[0]["surrogate_label"] == "ENN"
    assert recs[-1]["surrogate"] == "vecchia"
    assert recs[-1]["fit_seconds_mu"] == 0.7


def test_load_synthetic_sine_benchmark_json_dir(tmp_path: Path, capsys):
    nan = float("nan")
    r_a = _bench(
        enn=_br(0.1, 0.2, -1.0),
        smac_rf=_br(nan, nan, nan),
        dngo=_br(0.3, 0.4, -2.0),
        exact_gp=_br(0.5, 0.6, -3.0),
        svgp_default=_br(0.7, 0.8, -4.0),
        svgp_linear=_br(0.9, 1.0, -5.0),
        vecchia=_br(1.0, 1.1, -6.0),
    )
    z = _br(0.0, 0.0, 0.0)
    r_b = _bench(
        enn=_br(2.0, 0.0, 0.0),
        smac_rf=z,
        dngo=z,
        exact_gp=z,
        svgp_default=z,
        svgp_linear=z,
        vecchia=z,
    )
    sub = tmp_path / "exp"
    sub.mkdir()
    write_synthetic_sine_benchmark_json(
        sub / "b_second.json",
        synthetic_sine_benchmark_result_to_payload(r_b, n=2, d=2, function_name="ackley", problem_seed=1),
    )
    write_synthetic_sine_benchmark_json(
        sub / "a_first.json",
        synthetic_sine_benchmark_result_to_payload(r_a, n=10, d=3, function_name="sphere", problem_seed=7),
    )
    rows, benches = load_synthetic_sine_benchmark_json_dir(sub, verbose=True)
    assert [r["file"] for r in rows] == ["a_first.json", "b_second.json"]
    assert rows[0]["N"] == 10 and rows[0]["function_name"] == "sphere"
    assert rows[1]["N"] == 2 and rows[1]["function_name"] == "ackley"
    assert benches[0].results["enn"].fit_seconds.mu == r_a.results["enn"].fit_seconds.mu and math.isnan(benches[0].results["smac_rf"].fit_seconds.mu)
    assert benches[1].results["enn"].fit_seconds.mu == r_b.results["enn"].fit_seconds.mu
    assert "loaded 2 runs" in capsys.readouterr().out

    empty_d = tmp_path / "empty"
    empty_d.mkdir()
    empty_rows, _ = load_synthetic_sine_benchmark_json_dir(empty_d, verbose=False)
    assert empty_rows == []


def test_load_synthetic_sine_benchmark_json_dir_long(tmp_path: Path):
    nan = float("nan")
    r = _bench(
        enn=_br(0.1, 0.2, -1.0, se_f=0.01, se_n=0.02, se_l=0.03),
        smac_rf=_br(nan, nan, nan),
        dngo=_br(0.3, 0.4, -2.0),
        exact_gp=_br(0.5, 0.6, -3.0),
        svgp_default=_br(0.7, 0.8, -4.0),
        svgp_linear=_br(0.9, 1.0, -5.0),
        vecchia=_br(1.1, 1.2, -6.0),
    )
    sub = tmp_path / "exp"
    sub.mkdir()
    write_synthetic_sine_benchmark_json(
        sub / "a_first.json",
        synthetic_sine_benchmark_result_to_payload(r, n=10, d=3, function_name="sphere", problem_seed=7),
    )

    df = load_synthetic_sine_benchmark_json_dir_long(sub, verbose=False)

    assert len(df) == 7
    assert list(df.columns) == [
        "file",
        "N",
        "D",
        "function_name",
        "problem_seed",
        "num_reps",
        "surrogate",
        "surrogate_label",
        "fit_seconds_mu",
        "fit_seconds_se",
        "normalized_rmse_mu",
        "normalized_rmse_se",
        "log_likelihood_mu",
        "log_likelihood_se",
    ]
    enn_row = df[df["surrogate"] == "enn"].iloc[0]
    assert enn_row["fit_seconds_mu"] == 0.1
    assert enn_row["fit_seconds_se"] == 0.01
    assert enn_row["surrogate_label"] == "ENN"


def test_synthetic_sine_benchmark_json_file_round_trip(tmp_path: Path):
    inf = float("inf")
    r = _bench(
        enn=_br(1.0, 0.1, 0.0),
        smac_rf=_br(inf, 0.0, -1.0),
        dngo=_br(1.0, 0.1, 0.0),
        exact_gp=_br(1.0, 0.1, 0.0),
        svgp_default=_br(1.0, 0.1, 0.0),
        svgp_linear=_br(1.0, 0.1, 0.0),
        vecchia=_br(1.0, 0.1, 0.0),
    )
    p = synthetic_sine_benchmark_result_to_payload(r, n=5, d=2, function_name="sine", problem_seed=0)
    path = tmp_path / "x.json"
    write_synthetic_sine_benchmark_json(path, p)
    raw = path.read_text(encoding="utf-8")
    assert "Infinity" in raw or "inf" in raw.lower()
    r2, meta = read_synthetic_sine_benchmark_json(path)
    assert meta["function_name"] == "sine"
    assert math.isinf(r2.results["smac_rf"].fit_seconds.mu)


def test_synthetic_sine_benchmark_json_load_strips_unknown_keys():
    """Extra keys (forward compatibility) are ignored when building the dataclass."""
    from dataclasses import asdict

    z = _bench(
        enn=_br(0.0, 0.0, 0.0),
        smac_rf=_br(0.0, 0.0, 0.0),
        dngo=_br(0.0, 0.0, 0.0),
        exact_gp=_br(0.0, 0.0, 0.0),
        svgp_default=_br(0.0, 0.0, 0.0),
        svgp_linear=_br(0.0, 0.0, 0.0),
        vecchia=_br(0.0, 0.0, 0.0),
    )
    d = asdict(z)
    d[META_KEY] = {"N": 1, "D": 1, "function_name": None, "problem_seed": 0}
    d["future_field"] = 123.0
    bench, _meta = synthetic_sine_benchmark_from_payload(d)
    assert isinstance(bench, SyntheticSineSurrogateBenchmark)


@pytest.mark.parametrize(
    ("fn", "expected_token"),
    [("sine", "sine"), ("sphere", "sphere"), ("a/b", "a_b")],
)
def test_synthetic_sine_benchmark_config_slug(fn, expected_token):
    s = synthetic_sine_benchmark_config_slug(n=8, d=4, function_name=fn, problem_seed=2)
    assert s == f"N8_D4_{expected_token}_pseed2"
    s5 = synthetic_sine_benchmark_config_slug(n=8, d=4, function_name=fn, problem_seed=2, num_reps=5)
    assert s5 == f"N8_D4_{expected_token}_pseed2_nrep5"


def test_synthetic_sine_benchmark_rep_slug():
    s = synthetic_sine_benchmark_rep_slug(n=8, d=4, function_name="sphere", problem_seed=2, rep_index=3)
    assert s == "N8_D4_sphere_pseed2_rep3"


def test_payload_run_synthetic_sine_benchmark_modal_to_disk(monkeypatch, tmp_path: Path):
    import experiments.synthetic_sine_benchmark_payload as pl

    z = _br(0.0, 0.0, 0.0)
    zero = _bench(enn=z, smac_rf=z, dngo=z, exact_gp=z, svgp_default=z, svgp_linear=z, vecchia=z)

    class _App:
        def run(self):
            return contextlib.nullcontext()

    class _Rem:
        @staticmethod
        def remote(n, d, fn, ps, *_args):
            return pl.synthetic_sine_benchmark_result_to_payload(zero, n=n, d=d, function_name=fn, problem_seed=ps)

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
    @dataclass(frozen=True)
    class _J:
        n: int
        d: int
        target: str
        problem_seed: int = 0

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
    rep0 = _bench(
        enn=_br(1.0, 2.0, 3.0),
        smac_rf=_br(2.0, 3.0, 4.0),
        dngo=_br(3.0, 4.0, 5.0),
        exact_gp=_br(4.0, 5.0, 6.0),
        svgp_default=_br(5.0, 6.0, 7.0),
        svgp_linear=_br(6.0, 7.0, 8.0),
        vecchia=_br(7.0, 8.0, 9.0),
    )
    rep1 = _bench(
        enn=_br(3.0, 4.0, 5.0),
        smac_rf=_br(4.0, 5.0, 6.0),
        dngo=_br(5.0, 6.0, 7.0),
        exact_gp=_br(6.0, 7.0, 8.0),
        svgp_default=_br(7.0, 8.0, 9.0),
        svgp_linear=_br(8.0, 9.0, 10.0),
        vecchia=_br(9.0, 10.0, 11.0),
    )
    rep0_path = impl._rep_json_dest(out_dir, n=10, d=3, function_name="sphere", problem_seed=7, rep_index=0)
    rep1_path = impl._rep_json_dest(out_dir, n=10, d=3, function_name="sphere", problem_seed=7, rep_index=1)
    write_synthetic_sine_benchmark_json(
        rep0_path,
        synthetic_sine_benchmark_result_to_payload(rep0, n=10, d=3, function_name="sphere", problem_seed=7, num_reps=1),
    )
    write_synthetic_sine_benchmark_json(
        rep1_path,
        synthetic_sine_benchmark_result_to_payload(rep1, n=10, d=3, function_name="sphere", problem_seed=8, num_reps=1),
    )

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

    def fake_build(n, d, function_name, problem_seed, num_reps=1):
        captured["args"] = (n, d, function_name, problem_seed, num_reps)
        return {
            META_KEY: {
                "N": n,
                "D": d,
                "function_name": function_name,
                "problem_seed": problem_seed,
                "num_reps": num_reps,
            }
        }

    monkeypatch.setattr(impl.ssbp, "build_synthetic_sine_benchmark_remote_payload", fake_build)
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)

    impl.synthetic_sine_benchmark_batch_worker.info.raw_f(("tag-x", 12, 2, "sphere", 17, 3, 10))

    expected_data_seed = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=3)
    assert captured["args"] == (12, 2, "sphere", expected_data_seed, 1)
    payload, *_rest = store["N12_D2_sphere_pseed17_nrep10-rep3"]
    assert payload[META_KEY]["problem_seed"] == 17
    assert payload[META_KEY]["data_seed"] == expected_data_seed
    assert payload[META_KEY]["rep_index"] == 3


def test_batches_impl_submit_missing_prints_data_seed_ranges(monkeypatch, capsys):
    import experiments.modal_synthetic_sine_benchmark_batches_impl as impl

    spawned = []

    class _SpawnFn:
        def spawn(self, batch, tag):
            spawned.append((list(batch), tag))

    monkeypatch.setattr(
        impl.modal.Function,
        "from_name",
        lambda *_args, **_kwargs: _SpawnFn(),
    )
    monkeypatch.setattr(
        impl,
        "_iter_missing_jobs",
        lambda *_args, **_kwargs: iter(
            [
                ("k0", (12, 2, "sphere", 17, 0, 4)),
                ("k2", (12, 2, "sphere", 17, 2, 4)),
                ("k3", (12, 2, "sphere", 17, 3, 4)),
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
                ("k0", (12, 2, "sphere", 17, 0, 4)),
                ("k2", (12, 2, "sphere", 17, 2, 4)),
                ("k3", (12, 2, "sphere", 17, 3, 4)),
            ],
            "tag-x",
        )
    ]
    assert f"data_seed range N=12 D=2 fn=sphere pseed=17 reps=0-3 (3/4) seeds={seed0}-{seed3}" in out
    assert "submitted 3 jobs" in out


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
        return _bench(
            enn=_br(0.1, 0.2, -1.0),
            smac_rf=_br(nan, nan, nan),
            dngo=_br(0.3, 0.4, -2.0),
            exact_gp=_br(0.5, 0.6, -3.0),
            svgp_default=_br(0.7, 0.8, -4.0),
            svgp_linear=_br(0.9, 1.0, -5.0),
            vecchia=_br(nan, nan, nan),
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
