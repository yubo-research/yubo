import math

import numpy as np
import pytest
import torch

from analysis.fitting_time.evaluate import (
    SyntheticSineSurrogateBenchmark,
    benchmark_synthetic_sine_surrogates,
    normalized_rmse,
    predictive_gaussian_log_likelihood,
)
from analysis.fitting_time.fitting_time import fit_svgp_default, fit_svgp_linear


def test_normalized_rmse_zero_on_perfect_copy():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert normalized_rmse(y, y) == 0.0


def test_normalized_rmse_torch_and_mismatch():
    y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    yhat = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    assert normalized_rmse(y, yhat) == 0.0
    with pytest.raises(ValueError, match="mismatch"):
        normalized_rmse(y, yhat[:2])


def test_predictive_gaussian_log_likelihood_matches_independent_normal():
    n = 4
    y = np.zeros(n)
    mu = np.zeros(n)
    v = np.ones(n)
    ll = predictive_gaussian_log_likelihood(y, mu, v)
    expected = n * (-0.5 * math.log(2 * math.pi))
    assert ll == pytest.approx(expected)


def test_fit_svgp_default_posterior_on_original_y_scale():
    """Regression: SVGP means must be in ``train_y`` units (not z-scores)."""
    torch.manual_seed(0)
    n, d = 45, 3
    x = torch.rand(n, d, dtype=torch.float64)
    y = 5.0 + 2.0 * x[:, 0:1] + 0.5 * x[:, 1:2] + 0.15 * torch.randn(n, 1, dtype=torch.float64)
    _, y_hat, _ = fit_svgp_default(x[:35], y[:35], x[35:])
    m = float(y_hat.mean())
    assert m > 2.0 and m < 12.0, f"expected means near training level ~5–9, got {m}"


def test_fit_svgp_linear_runs_and_matches_y_scale():
    torch.manual_seed(1)
    n, d = 40, 2
    x = torch.rand(n, d, dtype=torch.float64)
    y = 3.0 + x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1, dtype=torch.float64)
    _, y_hat, _ = fit_svgp_linear(x[:30], y[:30], x[30:])
    assert float(y_hat.mean()) > 1.0 and float(y_hat.mean()) < 10.0


def test_predictive_gaussian_log_likelihood_torch():
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    mu = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    v = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    ll_t = predictive_gaussian_log_likelihood(y, mu, v)
    ll_np = predictive_gaussian_log_likelihood(y.numpy(), mu.numpy(), v.numpy())
    assert ll_t == pytest.approx(ll_np)


def test_synthetic_sine_surrogate_benchmark_print_table(capsys):
    r = SyntheticSineSurrogateBenchmark(
        enn_fit_seconds=0.01,
        enn_normalized_rmse=0.5,
        enn_log_likelihood=-10.0,
        smac_rf_fit_seconds=float("nan"),
        smac_rf_normalized_rmse=float("nan"),
        smac_rf_log_likelihood=float("nan"),
        dngo_fit_seconds=1.0,
        dngo_normalized_rmse=0.25,
        dngo_log_likelihood=-8.0,
        exact_gp_fit_seconds=2.0,
        exact_gp_normalized_rmse=0.1,
        exact_gp_log_likelihood=-7.0,
        svgp_default_fit_seconds=3.0,
        svgp_default_normalized_rmse=0.15,
        svgp_default_log_likelihood=-7.5,
        svgp_linear_fit_seconds=3.5,
        svgp_linear_normalized_rmse=0.14,
        svgp_linear_log_likelihood=-7.4,
    )
    r.print_table()
    out = capsys.readouterr().out
    assert "Surrogate" in out and "Speedup" in out and "NRMSE" in out and "LogLik" in out
    assert "ENN" in out and "SMAC RF" in out and "Exact GP" in out
    assert "SVGP_default" in out and "SVGP_linear" in out
    assert "nan" in out
    assert "100" in out  # DNGO 1.0s / ENN 0.01s


@pytest.mark.parametrize("function_name", [None, "sphere"])
def test_benchmark_synthetic_sine_surrogates_smoke(function_name):
    kwargs = {"N": 28, "D": 2}
    if function_name is not None:
        kwargs["function_name"] = function_name
    r = benchmark_synthetic_sine_surrogates(**kwargs)
    assert isinstance(r, SyntheticSineSurrogateBenchmark)
    assert math.isfinite(r.enn_fit_seconds) and math.isfinite(r.enn_normalized_rmse)
    assert math.isfinite(r.enn_log_likelihood)
    assert math.isfinite(r.dngo_fit_seconds) and math.isfinite(r.dngo_normalized_rmse)
    assert math.isfinite(r.dngo_log_likelihood)
    assert math.isfinite(r.exact_gp_fit_seconds) and math.isfinite(r.exact_gp_normalized_rmse)
    assert math.isfinite(r.exact_gp_log_likelihood)
    assert math.isfinite(r.svgp_default_fit_seconds) and math.isfinite(r.svgp_default_normalized_rmse)
    assert math.isfinite(r.svgp_default_log_likelihood)
    assert math.isfinite(r.svgp_linear_fit_seconds) and math.isfinite(r.svgp_linear_normalized_rmse)
    assert math.isfinite(r.svgp_linear_log_likelihood)
    try:
        import smac  # noqa: F401
    except ImportError:
        assert math.isnan(r.smac_rf_fit_seconds) and math.isnan(r.smac_rf_normalized_rmse)
        assert math.isnan(r.smac_rf_log_likelihood)
    else:
        assert math.isfinite(r.smac_rf_fit_seconds)
        assert math.isfinite(r.smac_rf_normalized_rmse)
        assert math.isfinite(r.smac_rf_log_likelihood)
