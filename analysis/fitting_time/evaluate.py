"""Metrics and end-to-end surrogate benchmarks (explicit synthetic targets)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# Explicit name for the FittingTime-style target: ``U(0,1)^D`` and ``mean(sin(2πx))+noise``.
SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME = "sine"

__all__ = [
    "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME",
    "SyntheticSineSurrogateBenchmark",
    "benchmark_synthetic_sine_surrogates",
    "draw_benchmark_synthetic_xy",
    "env_action_coords_to_surrogate_unit_x",
    "normalize_benchmark_function_name",
    "normalized_rmse",
    "predictive_gaussian_log_likelihood",
]


def env_action_coords_to_surrogate_unit_x(x: torch.Tensor) -> torch.Tensor:
    """Map env / benchmark action coordinates from ``[-1, 1]`` to BoTorch-style ``[0, 1]``.

    Pure-function envs and the synthetic sine target expose ``x`` in ``[-1, 1]``; GPs and
    other surrogates in this stack expect inputs in the unit cube.
    """
    return (x + 1.0) * 0.5


def normalize_benchmark_function_name(function_name: str) -> str:
    """Strip ``function_name`` and validate; canonicalize the unit-cube sine target to ``\"sine\"``."""
    if not isinstance(function_name, str):
        raise TypeError("function_name must be str")
    s = function_name.strip()
    if not s:
        raise ValueError("function_name must be non-empty (after strip)")
    if s.lower() == SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME:
        return SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME
    return s


def _as_float64_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def normalized_rmse(y_test, y_hat_test) -> float:
    """RMSE divided by the standard deviation of ``y_test`` (``ddof=0``).

    Accepts ``torch.Tensor`` or NumPy arrays; values are raveled to 1D.
    """
    yt = _as_float64_1d(y_test)
    yh = _as_float64_1d(y_hat_test)
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
    yt = _as_float64_1d(y_true)
    yh = _as_float64_1d(y_hat)
    pv = _as_float64_1d(pred_var)
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


def draw_benchmark_synthetic_xy(
    *,
    N: int,
    D: int,
    function_name: str,
    problem_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw train/test batches; ``x`` and ``x_test`` are in ``[-1, 1]`` (env / action scale)."""
    fn = normalize_benchmark_function_name(function_name)
    if fn == SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME:
        base = int(problem_seed)
        torch.manual_seed(base)
        x = torch.rand(N, D) * 2.0 - 1.0
        x_u = env_action_coords_to_surrogate_unit_x(x)
        y = torch.sin(2 * torch.pi * x_u).mean(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
        torch.manual_seed(base + 1)
        x_test = torch.rand(N, D) * 2.0 - 1.0
        x_test_u = env_action_coords_to_surrogate_unit_x(x_test)
        y_test = torch.sin(2 * torch.pi * x_test_u).mean(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
        return x, y, x_test, y_test
    from problems import pure_functions

    env_tag = f"f:{fn}-{D}d"
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
    except Exception:  # noqa: BLE001
        # SMAC is optional: missing deps, bad install, or runtime fit failure → degrade row.
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
    fit_vecchia,
) -> tuple[
    tuple[float, float, float],
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
    dt_vc, yh_vc, var_vc = fit_vecchia(train_x_t, train_y_t, x_test_t)
    nrmse_vc = normalized_rmse(y_test, yh_vc)
    ll_vc = predictive_gaussian_log_likelihood(y_test, yh_vc, var_vc)
    return (
        (dt_gp, nrmse_gp, ll_gp),
        (dt_svgp_d, nrmse_svgp_d, ll_svgp_d),
        (dt_svgp_l, nrmse_svgp_l, ll_svgp_l),
        (dt_vc, nrmse_vc, ll_vc),
    )


@dataclass(frozen=True)
class SyntheticSineSurrogateBenchmark:
    """Fit times (seconds) and normalized RMSE on test for each surrogate.

    Data is chosen by the required ``function_name`` passed to
    :func:`benchmark_synthetic_sine_surrogates`: use
    :data:`SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME` for ``FittingTime.ipynb``-style
    ``x ~ U(-1,1)^{N×D}`` with ``y = mean(sin(2π x_u), dim=1) + 0.1 ε`` where
    ``x_u = (x+1)/2`` (same distribution for ``x_u`` as the legacy ``U(0,1)`` draw); any
    other name uses ``f:{name}-{D}d`` from :mod:`problems.pure_functions` on
    ``U(-1,1)^{N×D}`` (same noise scale). Surrogates receive ``(x+1)/2`` in ``[0,1]`` via
    :func:`env_action_coords_to_surrogate_unit_x`.
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
    vecchia_fit_seconds: float
    vecchia_normalized_rmse: float
    vecchia_log_likelihood: float

    def print_table(self) -> None:
        """Print fit times (s), wall-clock ratio vs ENN (``t_surrogate / t_ENN``), NRMSE, log-likelihood (nats)."""

        def fmt(x: float) -> str:
            if math.isnan(x):
                return "nan"
            if math.isinf(x):
                return "inf" if x > 0 else "-inf"
            return f"{x:.6g}"

        def time_ratio_vs_enn(fit_sec: float) -> float:
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
        header = f"{'Surrogate':<{w_name}}  {'Fit (s)':>{w_sec}}  {'t/t_ENN':>{w_sp}}  {'NRMSE':>{w_rmse}}  {'LogLik':>{w_ll}}"
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
            (
                "Vecchia",
                self.vecchia_fit_seconds,
                self.vecchia_normalized_rmse,
                self.vecchia_log_likelihood,
            ),
        ]
        body = [
            f"{name:<{w_name}}  {fmt(sec):>{w_sec}}  {fmt(time_ratio_vs_enn(sec)):>{w_sp}}  {fmt(rmse):>{w_rmse}}  {fmt(ll):>{w_ll}}"
            for name, sec, rmse, ll in rows
        ]
        print("\n".join([header, sep, *body]))


def _synthetic_surrogate_benchmark_from_tensors(
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
) -> SyntheticSineSurrogateBenchmark:
    x_surr = env_action_coords_to_surrogate_unit_x(x)
    x_test_surr = env_action_coords_to_surrogate_unit_x(x_test)
    train_x = x_surr.detach().cpu().numpy().astype(np.float64)
    train_y = y.detach().cpu().numpy().astype(np.float64)
    x_test_np = x_test_surr.detach().cpu().numpy().astype(np.float64)
    (
        (dt_enn, nrmse_enn, ll_enn),
        (dt_smac, nrmse_smac, ll_smac),
        (dt_dngo, nrmse_dngo, ll_dngo),
    ) = _benchmark_numpy_surrogate_triples(train_x, train_y, x_test_np, y_test, fit_enn, fit_smac_rf, fit_dngo)
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
        vecchia_fit_seconds=dt_vc,
        vecchia_normalized_rmse=nrmse_vc,
        vecchia_log_likelihood=ll_vc,
    )


def benchmark_synthetic_sine_surrogates(
    *,
    N: int,
    D: int,
    function_name: str,
    problem_seed: int = 0,
) -> SyntheticSineSurrogateBenchmark:
    """Run ENN, SMAC RF, DNGO, exact GP, two SVGP variants, and RF Vecchia on synthetic data in ``D`` dims.

    ``function_name`` is **required** (non-empty after strip). Use
    :data:`SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME` (``\"sine\"``) for the FittingTime
    target: ``x ~ U(-1,1)^{N×D}``, ``y = mean(sin(2π (x+1)/2)) + 0.1 ε`` (equivalently
    ``x_u ~ U(0,1)`` with ``y = mean(sin(2π x_u)) + noise``).

    Any other name builds ``f"f:{function_name}-{D}d"`` via
    :mod:`problems.pure_functions`, draws ``x ~ U(-1,1)^{N×D}``, and sets ``y`` to the
    environment reward plus ``0.1 ε``. All surrogates are fit on ``(x+1)/2`` in
    ``[0,1]``; metrics still use the original ``y`` / ``y_test`` from the env draw.

    Reproducible RNG: for ``\"sine\"``, ``torch.manual_seed(problem_seed)`` before the
    train draw and ``torch.manual_seed(problem_seed + 1)`` before the test draw. For
    other names, the env uses ``problem_seed`` for distortion while train/test
    ``torch.rand`` use seeds ``0`` and ``1``.

    If the SMAC RF path is unavailable or raises (e.g. missing ``smac``, bad install,
    runtime fit failure), SMAC RF fields are set to ``nan``.

    **LogLik** (nats): ``sum_i log N(y_test_i | y_hat_i, v_i)``. ENN uses **epistemic**
    variance only (``PosteriorFlags(observation_noise=False)``, ``v_i = se_i^2``). SMAC RF
    uses forest variance plus ``0.1^2`` (synthetic observation noise).
    DNGO uses its full BLR predictive variance; exact GP, both SVGP variants, and
    Vecchia use GP predictive variance (Vecchia via ``pyvecch`` posterior diagonal).
    If ``pyvecch`` is missing or fit/predict fails, Vecchia fields are ``nan`` (same pattern as
    optional SMAC RF). On macOS, set ``YUBO_ALLOW_PYVECCH_ON_DARWIN=0`` to force-disable Vecchia
    if import still crashes in your environment.
    """
    from analysis.fitting_time.fitting_time import (
        fit_dngo,
        fit_enn,
        fit_exact_gp,
        fit_smac_rf,
        fit_svgp_default,
        fit_svgp_linear,
        fit_vecchia,
    )

    x, y, x_test, y_test = draw_benchmark_synthetic_xy(N=N, D=D, function_name=function_name, problem_seed=problem_seed)
    return _synthetic_surrogate_benchmark_from_tensors(
        x,
        y,
        x_test,
        y_test,
        fit_enn=fit_enn,
        fit_smac_rf=fit_smac_rf,
        fit_dngo=fit_dngo,
        fit_exact_gp=fit_exact_gp,
        fit_svgp_default=fit_svgp_default,
        fit_svgp_linear=fit_svgp_linear,
        fit_vecchia=fit_vecchia,
    )
