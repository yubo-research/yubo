"""Per-surrogate timing / NRMSE / log-likelihood triples for benchmarks."""

from __future__ import annotations

import math

import numpy as np
import torch

from .evaluate_metrics import normalized_rmse, predictive_gaussian_log_likelihood


def _benchmark_numpy_surrogate_triples(
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_test_np: np.ndarray,
    y_test: torch.Tensor,
    fit_enn,
    fit_smac_rf,
    fit_dngo,
    *,
    b_fast_only: bool = False,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    dt_enn, yh_enn, var_enn = fit_enn(train_x, train_y, x_test_np)
    nrmse_enn = normalized_rmse(y_test, yh_enn)
    ll_enn = predictive_gaussian_log_likelihood(y_test, yh_enn, var_enn)
    try:
        dt_smac, yh_smac, var_smac = fit_smac_rf(train_x, train_y.reshape(-1), x_test_np)
        nrmse_smac = normalized_rmse(y_test, yh_smac)
        ll_smac = predictive_gaussian_log_likelihood(y_test, yh_smac, var_smac)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, ArithmeticError):
        dt_smac, nrmse_smac, ll_smac = math.nan, math.nan, math.nan
    if b_fast_only:
        dt_dngo, nrmse_dngo, ll_dngo = math.nan, math.nan, math.nan
    else:
        dt_dngo, yh_dngo, var_dngo = fit_dngo(train_x, train_y, x_test_np)
        nrmse_dngo = normalized_rmse(y_test, yh_dngo)
        ll_dngo = predictive_gaussian_log_likelihood(y_test, yh_dngo, var_dngo)
    return (
        (dt_enn, nrmse_enn, ll_enn),
        (dt_smac, nrmse_smac, ll_smac),
        (dt_dngo, nrmse_dngo, ll_dngo),
    )


def _benchmark_torch_gp_triples(
    train_x_t: torch.Tensor,
    train_y_t: torch.Tensor,
    x_test_t: torch.Tensor,
    y_test: torch.Tensor,
    fit_exact_gp,
    fit_svgp_default,
    fit_svgp_linear,
    fit_vecchia,
    *,
    b_fast_only: bool = False,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    if b_fast_only:
        nan3 = (math.nan, math.nan, math.nan)
        return (nan3, nan3, nan3, nan3)
    dt_gp, yh_gp, var_gp = fit_exact_gp(train_x_t, train_y_t, x_test_t)
    nrmse_gp = normalized_rmse(y_test, yh_gp)
    ll_gp = predictive_gaussian_log_likelihood(y_test, yh_gp, var_gp)
    dt_svgp_d, yh_svgp_d, var_svgp_d = fit_svgp_default(train_x_t, train_y_t, x_test_t)
    nrmse_svgp_d = normalized_rmse(y_test, yh_svgp_d)
    ll_svgp_d = predictive_gaussian_log_likelihood(y_test, yh_svgp_d, var_svgp_d)
    dt_svgp_l, yh_svgp_l, var_svgp_l = fit_svgp_linear(train_x_t, train_y_t, x_test_t)
    nrmse_svgp_l = normalized_rmse(y_test, yh_svgp_l)
    ll_svgp_l = predictive_gaussian_log_likelihood(y_test, yh_svgp_l, var_svgp_l)
    dt_vc, yh_vc, var_vc = fit_vecchia(train_x_t, train_y_t, x_test_t)
    nrmse_vc = normalized_rmse(y_test, yh_vc)
    ll_vc = predictive_gaussian_log_likelihood(y_test, yh_vc, var_vc)
    return (
        (dt_gp, nrmse_gp, ll_gp),
        (dt_svgp_d, nrmse_svgp_d, ll_svgp_d),
        (dt_svgp_l, nrmse_svgp_l, ll_svgp_l),
        (dt_vc, nrmse_vc, ll_vc),
    )


def _surrogate_metric_triples_from_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    fit_enn,
    fit_smac_rf,
    fit_dngo,
    fit_exact_gp,
    fit_svgp_default,
    fit_svgp_linear,
    fit_vecchia,
    b_fast_only: bool = False,
) -> dict[str, tuple[float, float, float]]:
    from .evaluate_metrics import env_action_coords_to_surrogate_unit_x

    x_surr = env_action_coords_to_surrogate_unit_x(x)
    x_test_surr = env_action_coords_to_surrogate_unit_x(x_test)
    train_x = x_surr.detach().cpu().numpy().astype(np.float64)
    train_y = y.detach().cpu().numpy().astype(np.float64)
    x_test_np = x_test_surr.detach().cpu().numpy().astype(np.float64)
    (
        (dt_enn, nrmse_enn, ll_enn),
        (dt_smac, nrmse_smac, ll_smac),
        (dt_dngo, nrmse_dngo, ll_dngo),
    ) = _benchmark_numpy_surrogate_triples(
        train_x,
        train_y,
        x_test_np,
        y_test,
        fit_enn,
        fit_smac_rf,
        fit_dngo,
        b_fast_only=b_fast_only,
    )
    train_x_t = x_surr.to(dtype=torch.float64)
    train_y_t = y.to(dtype=torch.float64)
    x_test_t = x_test_surr.to(dtype=torch.float64)
    (
        (dt_gp, nrmse_gp, ll_gp),
        (dt_svgp_d, nrmse_svgp_d, ll_svgp_d),
        (dt_svgp_l, nrmse_svgp_l, ll_svgp_l),
        (dt_vc, nrmse_vc, ll_vc),
    ) = _benchmark_torch_gp_triples(
        train_x_t,
        train_y_t,
        x_test_t,
        y_test,
        fit_exact_gp,
        fit_svgp_default,
        fit_svgp_linear,
        fit_vecchia,
        b_fast_only=b_fast_only,
    )
    return {
        "enn": (dt_enn, nrmse_enn, ll_enn),
        "smac_rf": (dt_smac, nrmse_smac, ll_smac),
        "dngo": (dt_dngo, nrmse_dngo, ll_dngo),
        "exact_gp": (dt_gp, nrmse_gp, ll_gp),
        "svgp_default": (dt_svgp_d, nrmse_svgp_d, ll_svgp_d),
        "svgp_linear": (dt_svgp_l, nrmse_svgp_l, ll_svgp_l),
        "vecchia": (dt_vc, nrmse_vc, ll_vc),
    }


def aggregate_surrogate_replicates(rows: list[dict[str, tuple[float, float, float]]]):
    from .evaluate_class import SyntheticSineSurrogateBenchmark
    from .evaluate_metrics import SURROGATE_BENCHMARK_KEYS, BMResult, _mean_and_sem

    out: dict[str, BMResult] = {}
    for key in SURROGATE_BENCHMARK_KEYS:
        fs = [r[key][0] for r in rows]
        nrs = [r[key][1] for r in rows]
        lls = [r[key][2] for r in rows]
        out[key] = BMResult(
            fit_seconds=_mean_and_sem(fs),
            normalized_rmse=_mean_and_sem(nrs),
            log_likelihood=_mean_and_sem(lls),
        )
    return SyntheticSineSurrogateBenchmark(results=out)
