"""Metrics and end-to-end surrogate benchmarks (synthetic sine or ``f:`` pure benchmarks)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "SyntheticSineSurrogateBenchmark",
    "benchmark_synthetic_sine_surrogates",
    "normalized_rmse",
    "predictive_gaussian_log_likelihood",
]


def normalized_rmse(y_test, y_hat_test) -> float:
    """RMSE divided by the standard deviation of ``y_test`` (``ddof=0``).

    Accepts ``torch.Tensor`` or NumPy arrays; values are raveled to 1D.
    """
    yt = y_test.detach().cpu().numpy() if isinstance(y_test, torch.Tensor) else np.asarray(y_test, dtype=np.float64)
    yh = y_hat_test.detach().cpu().numpy() if isinstance(y_hat_test, torch.Tensor) else np.asarray(y_hat_test, dtype=np.float64)
    yt = np.asarray(yt, dtype=np.float64).reshape(-1)
    yh = np.asarray(yh, dtype=np.float64).reshape(-1)
    if yt.shape != yh.shape:
        raise ValueError(f"y_test and y_hat_test length mismatch: {yt.shape} vs {yh.shape}")
    rmse = float(np.sqrt(np.mean((yt - yh) ** 2)))
    scale = float(np.std(yt, ddof=0))
    if scale <= 1e-15:
        return 0.0 if rmse <= 1e-15 else math.inf
    return rmse / scale


def predictive_gaussian_log_likelihood(y_true, y_hat, pred_var) -> float:
    """Sum of independent Gaussian log-densities: ``sum_i log N(y_i | mu_i, v_i)``.

    ``y_true``, ``y_hat``, and ``pred_var`` may be ``torch.Tensor`` or NumPy arrays;
    all are raveled to 1D. ``pred_var`` must hold **per-point predictive variance**
    (heteroscedastic), strictly positive after an internal floor.

    Natural units (nats); larger is better.
    """
    yt = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true, dtype=np.float64)
    yh = y_hat.detach().cpu().numpy() if isinstance(y_hat, torch.Tensor) else np.asarray(y_hat, dtype=np.float64)
    pv = pred_var.detach().cpu().numpy() if isinstance(pred_var, torch.Tensor) else np.asarray(pred_var, dtype=np.float64)
    yt = np.asarray(yt, dtype=np.float64).reshape(-1)
    yh = np.asarray(yh, dtype=np.float64).reshape(-1)
    pv = np.asarray(pv, dtype=np.float64).reshape(-1)
    if yt.shape != yh.shape or yt.shape != pv.shape:
        raise ValueError(f"Shape mismatch for log-likelihood: y_true {yt.shape}, y_hat {yh.shape}, pred_var {pv.shape}")
    pv = np.maximum(pv, 1e-30)
    resid = yt - yh
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * pv) + resid**2 / pv)))


def _batch_pure_env_reward(env, actions_np: np.ndarray) -> np.ndarray:
    """Evaluate ``PureFunctionEnv`` rewards for each row of ``actions_np`` (N, D) in [-1, 1]."""
    n = actions_np.shape[0]
    y = np.empty((n, 1), dtype=np.float64)
    for i in range(n):
        step = env.step(actions_np[i])
        y[i, 0] = float(step.reward)
    return y


def _draw_benchmark_synthetic_xy(
    *,
    N: int,
    D: int,
    function_name: str | None,
    problem_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if function_name is None:
        torch.manual_seed(0)
        x = torch.rand(N, D)
        y = torch.sin(2 * torch.pi * x).mean(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
        torch.manual_seed(1)
        x_test = torch.rand(N, D)
        y_test = torch.sin(2 * torch.pi * x_test).mean(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
        return x, y, x_test, y_test
    from problems import pure_functions

    env_tag = f"f:{function_name}-{D}d"
    env = pure_functions.make(env_tag, problem_seed=problem_seed, distort=True)
    torch.manual_seed(0)
    x = torch.rand(N, D) * 2.0 - 1.0
    y_body = _batch_pure_env_reward(env, x.detach().cpu().numpy().astype(np.float64))
    y = torch.tensor(y_body, dtype=torch.float64) + 0.1 * torch.randn(N, 1)
    torch.manual_seed(1)
    x_test = torch.rand(N, D) * 2.0 - 1.0
    y_test_body = _batch_pure_env_reward(env, x_test.detach().cpu().numpy().astype(np.float64))
    y_test = torch.tensor(y_test_body, dtype=torch.float64) + 0.1 * torch.randn(N, 1)
    return x, y, x_test, y_test


def _benchmark_numpy_surrogate_triples(
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_test_np: np.ndarray,
    y_test: torch.Tensor,
    fit_enn,
    fit_smac_rf,
    fit_dngo,
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
    except ImportError:
        dt_smac, nrmse_smac, ll_smac = math.nan, math.nan, math.nan
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
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    dt_gp, yh_gp, var_gp = fit_exact_gp(train_x_t, train_y_t, x_test_t)
    nrmse_gp = normalized_rmse(y_test, yh_gp)
    ll_gp = predictive_gaussian_log_likelihood(y_test, yh_gp, var_gp)
    dt_svgp_d, yh_svgp_d, var_svgp_d = fit_svgp_default(train_x_t, train_y_t, x_test_t)
    nrmse_svgp_d = normalized_rmse(y_test, yh_svgp_d)
    ll_svgp_d = predictive_gaussian_log_likelihood(y_test, yh_svgp_d, var_svgp_d)
    dt_svgp_l, yh_svgp_l, var_svgp_l = fit_svgp_linear(train_x_t, train_y_t, x_test_t)
    nrmse_svgp_l = normalized_rmse(y_test, yh_svgp_l)
    ll_svgp_l = predictive_gaussian_log_likelihood(y_test, yh_svgp_l, var_svgp_l)
    return (
        (dt_gp, nrmse_gp, ll_gp),
        (dt_svgp_d, nrmse_svgp_d, ll_svgp_d),
        (dt_svgp_l, nrmse_svgp_l, ll_svgp_l),
    )


@dataclass(frozen=True)
class SyntheticSineSurrogateBenchmark:
    """Fit times (seconds) and normalized RMSE on test for each surrogate.

    Default data matches ``FittingTime.ipynb``: ``x ~ U(0,1)^{N×D}``,
    ``y = mean(sin(2πx), dim=1) + 0.1 ε``. With ``function_name`` set in
    :func:`benchmark_synthetic_sine_surrogates`, ``x`` is ``U(-1,1)^{N×D}`` and
    ``y`` comes from :mod:`problems.pure_functions` (same noise).
    """

    enn_fit_seconds: float
    enn_normalized_rmse: float
    enn_log_likelihood: float
    smac_rf_fit_seconds: float
    smac_rf_normalized_rmse: float
    smac_rf_log_likelihood: float
    dngo_fit_seconds: float
    dngo_normalized_rmse: float
    dngo_log_likelihood: float
    exact_gp_fit_seconds: float
    exact_gp_normalized_rmse: float
    exact_gp_log_likelihood: float
    svgp_default_fit_seconds: float
    svgp_default_normalized_rmse: float
    svgp_default_log_likelihood: float
    svgp_linear_fit_seconds: float
    svgp_linear_normalized_rmse: float
    svgp_linear_log_likelihood: float

    def print_table(self) -> None:
        """Print fit times (s), speedup vs ENN, NRMSE, and predictive log-likelihood (nats)."""

        def fmt(x: float) -> str:
            if math.isnan(x):
                return "nan"
            if math.isinf(x):
                return "inf" if x > 0 else "-inf"
            return f"{x:.6g}"

        def speedup(fit_sec: float) -> float:
            base = self.enn_fit_seconds
            if math.isnan(base) or math.isnan(fit_sec):
                return float("nan")
            if base <= 0.0:
                return 1.0 if fit_sec <= 0.0 else float("inf")
            return fit_sec / base

        w_name = 16
        w_sec = 14
        w_sp = 12
        w_rmse = 12
        w_ll = 14
        header = f"{'Surrogate':<{w_name}}  {'Fit (s)':>{w_sec}}  {'Speedup':>{w_sp}}  {'NRMSE':>{w_rmse}}  {'LogLik':>{w_ll}}"
        sep = "-" * len(header)
        rows: list[tuple[str, float, float, float]] = [
            (
                "ENN",
                self.enn_fit_seconds,
                self.enn_normalized_rmse,
                self.enn_log_likelihood,
            ),
            (
                "SMAC RF",
                self.smac_rf_fit_seconds,
                self.smac_rf_normalized_rmse,
                self.smac_rf_log_likelihood,
            ),
            (
                "DNGO",
                self.dngo_fit_seconds,
                self.dngo_normalized_rmse,
                self.dngo_log_likelihood,
            ),
            (
                "Exact GP",
                self.exact_gp_fit_seconds,
                self.exact_gp_normalized_rmse,
                self.exact_gp_log_likelihood,
            ),
            (
                "SVGP_default",
                self.svgp_default_fit_seconds,
                self.svgp_default_normalized_rmse,
                self.svgp_default_log_likelihood,
            ),
            (
                "SVGP_linear",
                self.svgp_linear_fit_seconds,
                self.svgp_linear_normalized_rmse,
                self.svgp_linear_log_likelihood,
            ),
        ]
        body = [f"{name:<{w_name}}  {fmt(sec):>{w_sec}}  {fmt(speedup(sec)):>{w_sp}}  {fmt(rmse):>{w_rmse}}  {fmt(ll):>{w_ll}}" for name, sec, rmse, ll in rows]
        print("\n".join([header, sep, *body]))


def benchmark_synthetic_sine_surrogates(
    *,
    N: int,
    D: int,
    function_name: str | None = None,
    problem_seed: int = 0,
) -> SyntheticSineSurrogateBenchmark:
    """Run ENN, SMAC RF, DNGO, exact GP, and two SVGP variants on synthetic data in ``D`` dims.

    If ``function_name`` is ``None`` (default), uses the FittingTime notebook target:
    ``x ~ U(0,1)^{N×D}``, ``y = mean(sin(2πx)) + 0.1 ε``.

    If ``function_name`` is set (e.g. ``"sphere"``, ``"ackley"``), builds the tag
    ``f"f:{function_name}-{D}d"`` as expected by
    :mod:`problems.pure_functions`, instantiates that benchmark via
    :func:`problems.pure_functions.make`, draws ``x ~ U(-1,1)^{N×D}`` as actions,
    sets ``y`` to the environment step reward plus ``0.1 ε`` (same noise scale as the
    sine default). Uses one env (``problem_seed`` for space distortion) for both train
    and test batches.

    Reproducible RNG: ``torch.manual_seed(0)`` for train, ``1`` for test.

    If ``smac`` is not installed, SMAC RF fields are set to ``nan``.

    **LogLik** (nats): ``sum_i log N(y_test_i | y_hat_i, v_i)``. ENN uses **epistemic**
    variance only (``PosteriorFlags(observation_noise=False)``, ``v_i = se_i^2``). SMAC RF
    uses forest variance plus ``0.1^2`` (synthetic observation noise).
    DNGO uses its full BLR predictive variance; exact GP and both SVGP variants use
    BoTorch posterior predictive variance (includes learned observation noise).
    """
    from analysis.fitting_time.fitting_time import (
        fit_dngo,
        fit_enn,
        fit_exact_gp,
        fit_smac_rf,
        fit_svgp_default,
        fit_svgp_linear,
    )

    x, y, x_test, y_test = _draw_benchmark_synthetic_xy(N=N, D=D, function_name=function_name, problem_seed=problem_seed)
    train_x = x.detach().cpu().numpy().astype(np.float64)
    train_y = y.detach().cpu().numpy().astype(np.float64)
    x_test_np = x_test.detach().cpu().numpy().astype(np.float64)
    (
        (dt_enn, nrmse_enn, ll_enn),
        (dt_smac, nrmse_smac, ll_smac),
        (dt_dngo, nrmse_dngo, ll_dngo),
    ) = _benchmark_numpy_surrogate_triples(train_x, train_y, x_test_np, y_test, fit_enn, fit_smac_rf, fit_dngo)
    train_x_t = x.to(dtype=torch.float64)
    train_y_t = y.to(dtype=torch.float64)
    x_test_t = x_test.to(dtype=torch.float64)
    (
        (dt_gp, nrmse_gp, ll_gp),
        (dt_svgp_d, nrmse_svgp_d, ll_svgp_d),
        (dt_svgp_l, nrmse_svgp_l, ll_svgp_l),
    ) = _benchmark_torch_gp_triples(
        train_x_t,
        train_y_t,
        x_test_t,
        y_test,
        fit_exact_gp,
        fit_svgp_default,
        fit_svgp_linear,
    )

    return SyntheticSineSurrogateBenchmark(
        enn_fit_seconds=dt_enn,
        enn_normalized_rmse=nrmse_enn,
        enn_log_likelihood=ll_enn,
        smac_rf_fit_seconds=dt_smac,
        smac_rf_normalized_rmse=nrmse_smac,
        smac_rf_log_likelihood=ll_smac,
        dngo_fit_seconds=dt_dngo,
        dngo_normalized_rmse=nrmse_dngo,
        dngo_log_likelihood=ll_dngo,
        exact_gp_fit_seconds=dt_gp,
        exact_gp_normalized_rmse=nrmse_gp,
        exact_gp_log_likelihood=ll_gp,
        svgp_default_fit_seconds=dt_svgp_d,
        svgp_default_normalized_rmse=nrmse_svgp_d,
        svgp_default_log_likelihood=ll_svgp_d,
        svgp_linear_fit_seconds=dt_svgp_l,
        svgp_linear_normalized_rmse=nrmse_svgp_l,
        svgp_linear_log_likelihood=ll_svgp_l,
    )
