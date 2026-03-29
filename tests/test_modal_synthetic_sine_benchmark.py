import contextlib
import math
from dataclasses import fields
from pathlib import Path

import pytest

from analysis.fitting_time.evaluate import SyntheticSineSurrogateBenchmark
from experiments.synthetic_sine_benchmark_payload import (
    META_KEY,
    build_synthetic_sine_benchmark_remote_payload,
    read_synthetic_sine_benchmark_json,
    run_synthetic_sine_benchmark_modal_to_disk,
    synthetic_sine_benchmark_config_slug,
    synthetic_sine_benchmark_from_payload,
    synthetic_sine_benchmark_result_to_payload,
    write_synthetic_sine_benchmark_json,
)


def test_synthetic_sine_benchmark_payload_round_trip():
    r = SyntheticSineSurrogateBenchmark(
        enn_fit_seconds=0.1,
        enn_normalized_rmse=0.2,
        enn_log_likelihood=-1.0,
        smac_rf_fit_seconds=float("nan"),
        smac_rf_normalized_rmse=float("nan"),
        smac_rf_log_likelihood=float("nan"),
        dngo_fit_seconds=0.3,
        dngo_normalized_rmse=0.4,
        dngo_log_likelihood=-2.0,
        exact_gp_fit_seconds=0.5,
        exact_gp_normalized_rmse=0.6,
        exact_gp_log_likelihood=-3.0,
        svgp_default_fit_seconds=0.7,
        svgp_default_normalized_rmse=0.8,
        svgp_default_log_likelihood=-4.0,
        svgp_linear_fit_seconds=0.9,
        svgp_linear_normalized_rmse=1.0,
        svgp_linear_log_likelihood=-5.0,
    )
    payload = synthetic_sine_benchmark_result_to_payload(r, n=10, d=3, function_name="sphere", problem_seed=7)
    r2, meta = synthetic_sine_benchmark_from_payload(payload)
    assert meta == {"N": 10, "D": 3, "function_name": "sphere", "problem_seed": 7}
    assert r2 == r
    assert math.isnan(r2.smac_rf_fit_seconds)


def test_synthetic_sine_benchmark_json_file_round_trip(tmp_path: Path):
    r = SyntheticSineSurrogateBenchmark(
        enn_fit_seconds=1.0,
        enn_normalized_rmse=0.1,
        enn_log_likelihood=0.0,
        smac_rf_fit_seconds=float("inf"),
        smac_rf_normalized_rmse=0.0,
        smac_rf_log_likelihood=-1.0,
        dngo_fit_seconds=1.0,
        dngo_normalized_rmse=0.1,
        dngo_log_likelihood=0.0,
        exact_gp_fit_seconds=1.0,
        exact_gp_normalized_rmse=0.1,
        exact_gp_log_likelihood=0.0,
        svgp_default_fit_seconds=1.0,
        svgp_default_normalized_rmse=0.1,
        svgp_default_log_likelihood=0.0,
        svgp_linear_fit_seconds=1.0,
        svgp_linear_normalized_rmse=0.1,
        svgp_linear_log_likelihood=0.0,
    )
    p = synthetic_sine_benchmark_result_to_payload(r, n=5, d=2, function_name=None, problem_seed=0)
    path = tmp_path / "x.json"
    write_synthetic_sine_benchmark_json(path, p)
    raw = path.read_text(encoding="utf-8")
    assert "Infinity" in raw or "inf" in raw.lower()
    r2, meta = read_synthetic_sine_benchmark_json(path)
    assert meta["function_name"] is None
    assert math.isinf(r2.smac_rf_fit_seconds)


def test_synthetic_sine_benchmark_json_load_strips_unknown_keys():
    """Extra keys (forward compatibility) are ignored when building the dataclass."""
    d = {f.name: 0.0 for f in fields(SyntheticSineSurrogateBenchmark)}
    d[META_KEY] = {"N": 1, "D": 1, "function_name": None, "problem_seed": 0}
    d["future_field"] = 123.0
    bench, _meta = synthetic_sine_benchmark_from_payload(d)
    assert isinstance(bench, SyntheticSineSurrogateBenchmark)


@pytest.mark.parametrize(
    ("fn", "expected_token"),
    [(None, "sine"), ("sphere", "sphere"), ("a/b", "a_b")],
)
def test_synthetic_sine_benchmark_config_slug(fn, expected_token):
    s = synthetic_sine_benchmark_config_slug(n=8, d=4, function_name=fn, problem_seed=2)
    assert s == f"N8_D4_{expected_token}_pseed2"


def test_payload_run_synthetic_sine_benchmark_modal_to_disk(monkeypatch, tmp_path: Path):
    import experiments.synthetic_sine_benchmark_payload as pl

    zero = SyntheticSineSurrogateBenchmark(
        enn_fit_seconds=0.0,
        enn_normalized_rmse=0.0,
        enn_log_likelihood=0.0,
        smac_rf_fit_seconds=0.0,
        smac_rf_normalized_rmse=0.0,
        smac_rf_log_likelihood=0.0,
        dngo_fit_seconds=0.0,
        dngo_normalized_rmse=0.0,
        dngo_log_likelihood=0.0,
        exact_gp_fit_seconds=0.0,
        exact_gp_normalized_rmse=0.0,
        exact_gp_log_likelihood=0.0,
        svgp_default_fit_seconds=0.0,
        svgp_default_normalized_rmse=0.0,
        svgp_default_log_likelihood=0.0,
        svgp_linear_fit_seconds=0.0,
        svgp_linear_normalized_rmse=0.0,
        svgp_linear_log_likelihood=0.0,
    )

    class _App:
        def run(self):
            return contextlib.nullcontext()

    class _Rem:
        @staticmethod
        def remote(n, d, fn, ps):
            return pl.synthetic_sine_benchmark_result_to_payload(zero, n=n, d=d, function_name=fn, problem_seed=ps)

    monkeypatch.setattr(pl.modal, "enable_output", lambda: contextlib.nullcontext())
    dest = run_synthetic_sine_benchmark_modal_to_disk(2, 2, "sphere", 3, tmp_path, app=_App(), remote_fn=_Rem())
    assert dest.exists() and "sphere" in dest.name
    dest_ws = run_synthetic_sine_benchmark_modal_to_disk(2, 2, "  \t", 4, tmp_path / "ws", app=_App(), remote_fn=_Rem())
    assert dest_ws.exists() and "sine" in dest_ws.name


def test_run_synthetic_sine_benchmark_modal_to_disk_none_function_name_is_default_sine(monkeypatch, tmp_path: Path):
    """``function_name=None`` must match empty string: default sine, not the literal ``\"None\"`` tag."""
    import experiments.modal_synthetic_sine_benchmark as msb

    class _StubRemote:
        @staticmethod
        def remote(n, d, fn, ps):
            assert fn is None
            return synthetic_sine_benchmark_result_to_payload(
                SyntheticSineSurrogateBenchmark(
                    enn_fit_seconds=1.0,
                    enn_normalized_rmse=0.1,
                    enn_log_likelihood=0.0,
                    smac_rf_fit_seconds=1.0,
                    smac_rf_normalized_rmse=0.1,
                    smac_rf_log_likelihood=0.0,
                    dngo_fit_seconds=1.0,
                    dngo_normalized_rmse=0.1,
                    dngo_log_likelihood=0.0,
                    exact_gp_fit_seconds=1.0,
                    exact_gp_normalized_rmse=0.1,
                    exact_gp_log_likelihood=0.0,
                    svgp_default_fit_seconds=1.0,
                    svgp_default_normalized_rmse=0.1,
                    svgp_default_log_likelihood=0.0,
                    svgp_linear_fit_seconds=1.0,
                    svgp_linear_normalized_rmse=0.1,
                    svgp_linear_log_likelihood=0.0,
                ),
                n=n,
                d=d,
                function_name=None,
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
    dest = msb.run_synthetic_sine_benchmark_modal_to_disk(3, 2, None, 9, tmp_path, remote_fn=_StubRemote())
    assert "sine" in dest.name and "None" not in dest.name
    r2, meta = read_synthetic_sine_benchmark_json(dest)
    assert meta["function_name"] is None
    assert r2.enn_fit_seconds == 1.0


def test_run_synthetic_sine_benchmark_modal_to_disk_mocked(monkeypatch, tmp_path: Path, capsys):
    import experiments.modal_synthetic_sine_benchmark as msb

    sample = synthetic_sine_benchmark_result_to_payload(
        SyntheticSineSurrogateBenchmark(
            enn_fit_seconds=1.0,
            enn_normalized_rmse=0.1,
            enn_log_likelihood=0.0,
            smac_rf_fit_seconds=1.0,
            smac_rf_normalized_rmse=0.1,
            smac_rf_log_likelihood=0.0,
            dngo_fit_seconds=1.0,
            dngo_normalized_rmse=0.1,
            dngo_log_likelihood=0.0,
            exact_gp_fit_seconds=1.0,
            exact_gp_normalized_rmse=0.1,
            exact_gp_log_likelihood=0.0,
            svgp_default_fit_seconds=1.0,
            svgp_default_normalized_rmse=0.1,
            svgp_default_log_likelihood=0.0,
            svgp_linear_fit_seconds=1.0,
            svgp_linear_normalized_rmse=0.1,
            svgp_linear_log_likelihood=0.0,
        ),
        n=3,
        d=2,
        function_name=None,
        problem_seed=9,
    )

    class _StubRemote:
        @staticmethod
        def remote(n, d, fn, ps):
            assert n == 3 and d == 2 and fn is None and ps == 9
            return sample

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.modal.enable_output",
        lambda: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "experiments.modal_synthetic_sine_benchmark.app.run",
        lambda: contextlib.nullcontext(),
    )
    dest = msb.run_synthetic_sine_benchmark_modal_to_disk(3, 2, "", 9, tmp_path, remote_fn=_StubRemote())
    assert dest.exists()
    r2, meta = read_synthetic_sine_benchmark_json(dest)
    assert meta["N"] == 3
    assert r2.enn_fit_seconds == 1.0


def test_main_prints_destination(monkeypatch, tmp_path: Path, capsys):
    import experiments.modal_synthetic_sine_benchmark as msb

    def _stub_modal_to_disk(n, d, function_name, problem_seed, output_dir, *, remote_fn=None):
        dest = Path(output_dir) / "stub.json"
        write_synthetic_sine_benchmark_json(
            dest,
            synthetic_sine_benchmark_result_to_payload(
                SyntheticSineSurrogateBenchmark(
                    enn_fit_seconds=0.5,
                    enn_normalized_rmse=0.1,
                    enn_log_likelihood=0.0,
                    smac_rf_fit_seconds=0.5,
                    smac_rf_normalized_rmse=0.1,
                    smac_rf_log_likelihood=0.0,
                    dngo_fit_seconds=0.5,
                    dngo_normalized_rmse=0.1,
                    dngo_log_likelihood=0.0,
                    exact_gp_fit_seconds=0.5,
                    exact_gp_normalized_rmse=0.1,
                    exact_gp_log_likelihood=0.0,
                    svgp_default_fit_seconds=0.5,
                    svgp_default_normalized_rmse=0.1,
                    svgp_default_log_likelihood=0.0,
                    svgp_linear_fit_seconds=0.5,
                    svgp_linear_normalized_rmse=0.1,
                    svgp_linear_log_likelihood=0.0,
                ),
                n=n,
                d=d,
                function_name=None if not str(function_name).strip() else str(function_name).strip(),
                problem_seed=problem_seed,
            ),
        )
        return dest

    monkeypatch.setattr(
        "experiments.modal_synthetic_sine_benchmark.run_synthetic_sine_benchmark_modal_to_disk",
        _stub_modal_to_disk,
    )
    msb.main(n=4, d=2, function_name="", problem_seed=1, output_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "wrote" in out and "stub.json" in out


def test_main_raw_f_invokes_modal_to_disk(monkeypatch, tmp_path: Path, capsys):
    """Kiss/coverage: exercise undecorated ``main`` (``main.info.raw_f``) without Modal cloud."""
    import experiments.modal_synthetic_sine_benchmark as msb

    def fake_disk(n, d, fn, ps, od):
        p = Path(od) / "kiss_main.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return p

    monkeypatch.setattr(msb, "run_synthetic_sine_benchmark_modal_to_disk", fake_disk)
    msb.main.info.raw_f(2, 3, "", 1, str(tmp_path))
    assert "wrote" in capsys.readouterr().out


def test_build_synthetic_sine_benchmark_remote_payload_delegates(monkeypatch):
    """Heavy ``benchmark_synthetic_sine_surrogates`` smoke lives in ``test_evaluate``; mock here."""
    captured: dict = {}

    def _fake_benchmark(*, N, D, function_name, problem_seed):
        captured["N"] = N
        captured["D"] = D
        captured["function_name"] = function_name
        captured["problem_seed"] = problem_seed
        return SyntheticSineSurrogateBenchmark(
            enn_fit_seconds=0.1,
            enn_normalized_rmse=0.2,
            enn_log_likelihood=-1.0,
            smac_rf_fit_seconds=float("nan"),
            smac_rf_normalized_rmse=float("nan"),
            smac_rf_log_likelihood=float("nan"),
            dngo_fit_seconds=0.3,
            dngo_normalized_rmse=0.4,
            dngo_log_likelihood=-2.0,
            exact_gp_fit_seconds=0.5,
            exact_gp_normalized_rmse=0.6,
            exact_gp_log_likelihood=-3.0,
            svgp_default_fit_seconds=0.7,
            svgp_default_normalized_rmse=0.8,
            svgp_default_log_likelihood=-4.0,
            svgp_linear_fit_seconds=0.9,
            svgp_linear_normalized_rmse=1.0,
            svgp_linear_log_likelihood=-5.0,
        )

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.benchmark_synthetic_sine_surrogates",
        _fake_benchmark,
    )
    payload = build_synthetic_sine_benchmark_remote_payload(11, 3, "ackley", 42)
    bench, meta = synthetic_sine_benchmark_from_payload(payload)
    assert captured == {"N": 11, "D": 3, "function_name": "ackley", "problem_seed": 42}
    assert bench.enn_fit_seconds == 0.1
    assert meta["function_name"] == "ackley"
