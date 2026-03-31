import contextlib
import math
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from analysis.fitting_time.batch_jobs import SyntheticBenchJob, example_two_targets_n12_d2
from analysis.fitting_time.evaluate import SURROGATE_BENCHMARK_KEYS, BMResult, MuSe, SyntheticSineSurrogateBenchmark
from experiments.synthetic_sine_benchmark_payload import (
    META_KEY,
    build_synthetic_sine_benchmark_remote_payload,
    load_synthetic_sine_benchmark_jobs,
    load_synthetic_sine_benchmark_json_dir,
    read_synthetic_sine_benchmark_json,
    run_synthetic_sine_benchmark_modal_to_disk,
    synthetic_sine_benchmark_config_slug,
    synthetic_sine_benchmark_from_payload,
    synthetic_sine_benchmark_result_to_payload,
    synthetic_surrogate_benchmark_row_caption,
    wide_surrogate_benchmark_row_to_comparison_records,
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
    assert meta == {"N": 10, "D": 3, "function_name": "sphere", "problem_seed": 7, "num_reps": 1}
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


def test_run_synthetic_sine_benchmark_modal_to_disk_persists_sine_function_name(monkeypatch, tmp_path: Path):
    """Filename slug and JSON meta use canonical ``\"sine\"`` (not ``None``)."""
    import experiments.modal_synthetic_sine_benchmark as msb

    o = _br(1.0, 0.1, 0.0)

    class _StubRemote:
        @staticmethod
        def remote(n, d, fn, ps, *_args):
            assert fn == "sine"
            return synthetic_sine_benchmark_result_to_payload(
                _bench(enn=o, smac_rf=o, dngo=o, exact_gp=o, svgp_default=o, svgp_linear=o, vecchia=o),
                n=n,
                d=d,
                function_name="sine",
                problem_seed=ps,
            )

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.modal.enable_output",
        lambda: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "experiments.modal_synthetic_sine_benchmark.app.run",
        lambda: contextlib.nullcontext(),
    )
    dest = msb.run_synthetic_sine_benchmark_modal_to_disk(3, 2, "sine", 9, tmp_path, remote_fn=_StubRemote())
    assert "sine" in dest.name and "None" not in dest.name
    r2, meta = read_synthetic_sine_benchmark_json(dest)
    assert meta["function_name"] == "sine"
    assert r2.results["enn"].fit_seconds.mu == 1.0


def test_run_synthetic_sine_benchmark_modal_to_disk_mocked(monkeypatch, tmp_path: Path, capsys):
    import experiments.modal_synthetic_sine_benchmark as msb

    o = _br(1.0, 0.1, 0.0)
    sample = synthetic_sine_benchmark_result_to_payload(
        _bench(enn=o, smac_rf=o, dngo=o, exact_gp=o, svgp_default=o, svgp_linear=o, vecchia=o),
        n=3,
        d=2,
        function_name="sine",
        problem_seed=9,
    )

    class _StubRemote:
        @staticmethod
        def remote(n, d, fn, ps, *_args):
            assert n == 3 and d == 2 and fn == "sine" and ps == 9
            return sample

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.modal.enable_output",
        lambda: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "experiments.modal_synthetic_sine_benchmark.app.run",
        lambda: contextlib.nullcontext(),
    )
    dest = msb.run_synthetic_sine_benchmark_modal_to_disk(3, 2, "sine", 9, tmp_path, remote_fn=_StubRemote())
    assert dest.exists()
    r2, meta = read_synthetic_sine_benchmark_json(dest)
    assert meta["N"] == 3
    assert r2.results["enn"].fit_seconds.mu == 1.0


def test_main_prints_destination(monkeypatch, tmp_path: Path, capsys):
    import experiments.modal_synthetic_sine_benchmark as msb

    def _stub_modal_to_disk(n, d, function_name, problem_seed, output_dir, *, remote_fn=None, **kwargs):
        dest = Path(output_dir) / "stub.json"
        h = _br(0.5, 0.1, 0.0)
        write_synthetic_sine_benchmark_json(
            dest,
            synthetic_sine_benchmark_result_to_payload(
                _bench(enn=h, smac_rf=h, dngo=h, exact_gp=h, svgp_default=h, svgp_linear=h, vecchia=h),
                n=n,
                d=d,
                function_name=function_name,
                problem_seed=problem_seed,
            ),
        )
        return dest

    monkeypatch.setattr(
        "experiments.modal_synthetic_sine_benchmark.run_synthetic_sine_benchmark_modal_to_disk",
        _stub_modal_to_disk,
    )
    msb.main(target="sine", n=4, d=2, problem_seed=1, output_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "wrote" in out and "stub.json" in out


def test_main_raw_f_invokes_modal_to_disk(monkeypatch, tmp_path: Path, capsys):
    """Kiss/coverage: exercise undecorated ``main`` (``main.info.raw_f``) without Modal cloud."""
    import experiments.modal_synthetic_sine_benchmark as msb

    def fake_disk(n, d, fn, ps, od, **kwargs):
        p = Path(od) / "kiss_main.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return p

    monkeypatch.setattr(msb, "run_synthetic_sine_benchmark_modal_to_disk", fake_disk)
    msb.main.info.raw_f("sine", 2, 3, 1, str(tmp_path))
    assert "wrote" in capsys.readouterr().out


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


def test_batch_entrypoint_runs_each_job(monkeypatch, tmp_path: Path, capsys):
    import experiments.modal_synthetic_sine_benchmark as msb

    calls: list[tuple[int, int, str, int]] = []

    def fake_disk(n, d, fn, ps, od, **kwargs):
        calls.append((n, d, fn, int(ps)))
        out = tmp_path / f"{n}_{fn}.json"
        out.write_text("{}")
        return out

    monkeypatch.setattr(msb, "run_synthetic_sine_benchmark_modal_to_disk", fake_disk)
    msb.batch.info.raw_f("example_two_targets_n12_d2", str(tmp_path / "out"))
    assert calls == [(12, 2, "sphere", 0), (12, 2, "sine", 0)]
    out = capsys.readouterr().out
    assert out.count("wrote") == 2


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
