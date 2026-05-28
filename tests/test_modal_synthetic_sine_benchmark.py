import math
from pathlib import Path

import pytest

from analysis.fitting_time.evaluate import SyntheticSineSurrogateBenchmark
from experiments.synthetic_sine_benchmark_payload import (
    META_KEY,
    load_synthetic_sine_benchmark_json_dir,
    load_synthetic_sine_benchmark_json_dir_long,
    read_synthetic_sine_benchmark_json,
    synthetic_sine_benchmark_config_slug,
    synthetic_sine_benchmark_from_payload,
    synthetic_sine_benchmark_rep_slug,
    synthetic_sine_benchmark_result_to_payload,
    synthetic_surrogate_benchmark_row_caption,
    wide_surrogate_benchmark_row_to_comparison_records,
    wide_surrogate_benchmark_row_to_long_records,
    write_synthetic_sine_benchmark_json,
)
from tests.synthetic_sine_benchmark_helpers import bench_result, make_surrogate_benchmark


def test_synthetic_sine_benchmark_payload_round_trip():
    nan = float("nan")
    r = make_surrogate_benchmark(
        enn=bench_result(0.1, 0.2, -1.0),
        smac_rf=bench_result(nan, nan, nan),
        dngo=bench_result(0.3, 0.4, -2.0),
        exact_gp=bench_result(0.5, 0.6, -3.0),
        svgp_default=bench_result(0.7, 0.8, -4.0),
        svgp_linear=bench_result(0.9, 1.0, -5.0),
        vecchia=bench_result(1.1, 1.2, -6.0),
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
    assert len(recs) == 8
    assert recs[0]["Surrogate"] == "ENN" and recs[0]["Fit (s) μ"] == 0.1
    assert math.isnan(recs[2]["Fit (s) μ"])
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
    assert len(recs) == 8
    assert recs[0]["surrogate"] == "enn"
    assert recs[0]["fit_seconds_mu"] == 0.1
    assert recs[0]["surrogate_label"] == "ENN"
    assert recs[-1]["surrogate"] == "vecchia"
    assert recs[-1]["fit_seconds_mu"] == 0.7


def test_load_synthetic_sine_benchmark_json_dir(tmp_path: Path, capsys):
    nan = float("nan")
    r_a = make_surrogate_benchmark(
        enn=bench_result(0.1, 0.2, -1.0),
        smac_rf=bench_result(nan, nan, nan),
        dngo=bench_result(0.3, 0.4, -2.0),
        exact_gp=bench_result(0.5, 0.6, -3.0),
        svgp_default=bench_result(0.7, 0.8, -4.0),
        svgp_linear=bench_result(0.9, 1.0, -5.0),
        vecchia=bench_result(1.0, 1.1, -6.0),
    )
    z = bench_result(0.0, 0.0, 0.0)
    r_b = make_surrogate_benchmark(
        enn=bench_result(2.0, 0.0, 0.0),
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
    r = make_surrogate_benchmark(
        enn=bench_result(0.1, 0.2, -1.0, se_f=0.01, se_n=0.02, se_l=0.03),
        smac_rf=bench_result(nan, nan, nan),
        dngo=bench_result(0.3, 0.4, -2.0),
        exact_gp=bench_result(0.5, 0.6, -3.0),
        svgp_default=bench_result(0.7, 0.8, -4.0),
        svgp_linear=bench_result(0.9, 1.0, -5.0),
        vecchia=bench_result(1.1, 1.2, -6.0),
    )
    sub = tmp_path / "exp"
    sub.mkdir()
    write_synthetic_sine_benchmark_json(
        sub / "a_first.json",
        synthetic_sine_benchmark_result_to_payload(r, n=10, d=3, function_name="sphere", problem_seed=7),
    )

    df = load_synthetic_sine_benchmark_json_dir_long(sub, verbose=False)

    assert len(df) == 8
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
    r = make_surrogate_benchmark(
        enn=bench_result(1.0, 0.1, 0.0),
        smac_rf=bench_result(inf, 0.0, -1.0),
        dngo=bench_result(1.0, 0.1, 0.0),
        exact_gp=bench_result(1.0, 0.1, 0.0),
        svgp_default=bench_result(1.0, 0.1, 0.0),
        svgp_linear=bench_result(1.0, 0.1, 0.0),
        vecchia=bench_result(1.0, 0.1, 0.0),
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

    z = make_surrogate_benchmark(
        enn=bench_result(0.0, 0.0, 0.0),
        smac_rf=bench_result(0.0, 0.0, 0.0),
        dngo=bench_result(0.0, 0.0, 0.0),
        exact_gp=bench_result(0.0, 0.0, 0.0),
        svgp_default=bench_result(0.0, 0.0, 0.0),
        svgp_linear=bench_result(0.0, 0.0, 0.0),
        vecchia=bench_result(0.0, 0.0, 0.0),
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
