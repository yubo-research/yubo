"""Metrics and end-to-end surrogate benchmarks (explicit synthetic targets)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# Explicit name for the FittingTime-style target: ``U(0,1)^D`` and ``mean(sin(2πx))+noise``.
SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME = "sine"

__all__ = [
    "BMResult",
    "MuSe",
    "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME",
    "SURROGATE_BENCHMARK_ROWS",
    "SURROGATE_BENCHMARK_KEYS",
    "SyntheticSineSurrogateBenchmark",
    "benchmark_synthetic_sine_surrogates",
    "draw_benchmark_synthetic_xy",
    "env_action_coords_to_surrogate_unit_x",
    "normalize_benchmark_function_name",
    "normalized_rmse",
    "predictive_gaussian_log_likelihood",
]

SURROGATE_BENCHMARK_KEYS: tuple[str, ...] = (
    "enn",
    "smac_rf",
    "dngo",
    "exact_gp",
    "svgp_default",
    "svgp_linear",
    "vecchia",
)

SURROGATE_BENCHMARK_ROWS: tuple[tuple[str, str], ...] = (
    ("enn", "ENN"),
    ("smac_rf", "SMAC RF"),
    ("dngo", "DNGO"),
    ("exact_gp", "Exact GP"),
    ("svgp_default", "SVGP_default"),
    ("svgp_linear", "SVGP_linear"),
    ("vecchia", "Vecchia"),
)


@dataclass(frozen=True)
class MuSe:
    """Mean and standard error of the mean (``se`` = sample std / sqrt(n), ``ddof=1``; ``se=0`` when ``n<=1``)."""

    mu: float
    se: float


@dataclass(frozen=True)
class BMResult:
    fit_seconds: MuSe
    normalized_rmse: MuSe
    log_likelihood: MuSe


def _mean_and_sem(values: list[float]) -> MuSe:
    a = np.asarray(values, dtype=np.float64)
    fin = np.isfinite(a)
    if int(fin.sum()) == 0:
        return MuSe(float("nan"), float("nan"))
    v = a[fin]
    n = int(v.size)
    mu = float(np.mean(v))
    if n <= 1:
        return MuSe(mu, 0.0)
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    return MuSe(mu, se)


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
    except Exception:  # noqa: BLE001
        # SMAC is optional: missing deps, bad install, or runtime fit failure → degrade row.
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


@dataclass(frozen=True)
class SyntheticSineSurrogateBenchmark:
    """Per-surrogate :class:`BMResult` (mean and SEM over replicates).

    Data is chosen by the required ``function_name`` passed to
    :func:`benchmark_synthetic_sine_surrogates`: use
    :data:`SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME` for ``FittingTime.ipynb``-style
    ``x ~ U(-1,1)^{N×D}`` with ``y = mean(sin(2π x_u), dim=1) + 0.1 ε`` where
    ``x_u = (x+1)/2`` (same distribution for ``x_u`` as the legacy ``U(0,1)`` draw); any
    other name uses ``f:{name}-{D}d`` from :mod:`problems.pure_functions` on
    ``U(-1,1)^{N×D}`` (same noise scale). Surrogates receive ``(x+1)/2`` in ``[0,1]`` via
    :func:`env_action_coords_to_surrogate_unit_x`.

    Keys are :data:`SURROGATE_BENCHMARK_KEYS`.
    """

    results: dict[str, BMResult]

    def __post_init__(self) -> None:
        got = frozenset(self.results)
        want = frozenset(SURROGATE_BENCHMARK_KEYS)
        if got != want:
            raise ValueError(f"results keys must match SURROGATE_BENCHMARK_KEYS exactly; missing {want - got}, extra {got - want}")

    def print_table(self) -> None:
        """Print a compact timing table: ``Fit t (s)``, ``t/t_ENN``, ``NRMSE``, ``LogLik`` as ``μ ± SE``."""

        def _nonfinite_token(x: float) -> str | None:
            if math.isnan(x):
                return "nan"
            if math.isinf(x):
                return "inf" if x > 0 else "-inf"
            return None

        def _fmt_mu_seconds(x: float) -> str:
            t = _nonfinite_token(x)
            if t is not None:
                return t
            if x == 0.0:
                return "0"
            if abs(x) < 0.1:
                s = f"{x:.6f}".rstrip("0").rstrip(".")
                return s if s not in ("", "-") else "0"
            return f"{x:.6g}"

        def _fmt_mu_nrmse(x: float) -> str:
            t = _nonfinite_token(x)
            if t is not None:
                return t
            return f"{x:.6g}"

        def _fmt_mu_loglik(x: float) -> str:
            t = _nonfinite_token(x)
            if t is not None:
                return t
            if abs(x) >= 10.0:
                return f"{x:.2f}"
            return f"{x:.6g}"

        def _fmt_se(x: float) -> str:
            t = _nonfinite_token(x)
            if t is not None:
                return t
            return f"{x:.2g}"

        def _fmt_se_loglik(x: float) -> str:
            t = _nonfinite_token(x)
            if t is not None:
                return t
            if abs(x) >= 1.0:
                return f"{x:.0f}"
            return f"{x:.2g}"

        def _fmt_ratio(x: float) -> str:
            t = _nonfinite_token(x)
            if t is not None:
                return t
            r = round(x)
            if abs(x - r) < 1e-5:
                return str(int(r))
            return f"{x:.1f}"

        def _pm_column(
            mus: list[float],
            ses: list[float],
            fmt_mu,
            fmt_se_fn=_fmt_se,
        ) -> list[str]:
            mu_strs = [fmt_mu(mu) for mu in mus]
            w_m = max(len(s) for s in mu_strs)
            out: list[str] = []
            for mu, mstr, se in zip(mus, mu_strs, ses, strict=True):
                if math.isnan(mu):
                    out.append("nan")
                    continue
                if math.isinf(mu):
                    out.append(mstr)
                    continue
                if not math.isfinite(se) or se <= 0.0:
                    out.append(f"{mstr:<{w_m}}")
                else:
                    out.append(f"{mstr:<{w_m}} ± {fmt_se_fn(se)}")
            return out

        def time_ratio_vs_enn(fit_mu: float) -> float:
            base = self.results["enn"].fit_seconds.mu
            if math.isnan(base) or math.isnan(fit_mu):
                return float("nan")
            if base <= 0.0:
                return 1.0 if fit_mu <= 0.0 else float("inf")
            return fit_mu / base

        rows = [self.results[k] for k, _ in SURROGATE_BENCHMARK_ROWS]
        fit_m = [br.fit_seconds.mu for br in rows]
        fit_s = [br.fit_seconds.se for br in rows]
        nrm_m = [br.normalized_rmse.mu for br in rows]
        nrm_s = [br.normalized_rmse.se for br in rows]
        ll_m = [br.log_likelihood.mu for br in rows]
        ll_s = [br.log_likelihood.se for br in rows]
        ratio_m = [time_ratio_vs_enn(br.fit_seconds.mu) for br in rows]

        fit_cells = _pm_column(fit_m, fit_s, _fmt_mu_seconds)
        nrmse_cells = _pm_column(nrm_m, nrm_s, _fmt_mu_nrmse)
        ll_cells = _pm_column(ll_m, ll_s, _fmt_mu_loglik, fmt_se_fn=_fmt_se_loglik)
        ratio_cells = [_fmt_ratio(x) for x in ratio_m]

        w_name = max(len("Surrogate"), max(len(n) for _k, n in SURROGATE_BENCHMARK_ROWS))
        h_fit = "Fit t (s)"
        h_nrmse = "NRMSE"
        h_ll = "LogLik"
        h_sp = "t/t_ENN"
        w_fit = max(len(h_fit), max(len(c) for c in fit_cells))
        w_sp = max(len(h_sp), max(len(c) for c in ratio_cells))
        w_n = max(len(h_nrmse), max(len(c) for c in nrmse_cells))
        w_ll = max(len(h_ll), max(len(c) for c in ll_cells))
        header = f"{'Surrogate':<{w_name}}  {h_fit:>{w_fit}}  {h_sp:>{w_sp}}  {h_nrmse:>{w_n}}  {h_ll:>{w_ll}}"
        sep = "-" * len(header)
        body: list[str] = []
        for i, (_key, name) in enumerate(SURROGATE_BENCHMARK_ROWS):
            body.append(f"{name:<{w_name}}  {fit_cells[i]:>{w_fit}}  {ratio_cells[i]:>{w_sp}}  {nrmse_cells[i]:>{w_n}}  {ll_cells[i]:>{w_ll}}")
        print("\n".join([header, sep, *body]))


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


def _aggregate_surrogate_replicates(rows: list[dict[str, tuple[float, float, float]]]) -> SyntheticSineSurrogateBenchmark:
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


def benchmark_synthetic_sine_surrogates(
    *,
    N: int,
    D: int,
    function_name: str,
    problem_seed: int = 0,
    num_reps: int = 1,
    b_fast_only: bool = False,
) -> SyntheticSineSurrogateBenchmark:
    """Run ENN, SMAC RF, DNGO, exact GP, two SVGP variants, and RF Vecchia on synthetic data in ``D`` dims.

    If ``b_fast_only`` is true, only ENN and SMAC RF are fit; DNGO, exact GP, both SVGPs, and Vecchia
    are skipped and their metrics are set to ``nan`` (same as an unavailable surrogate row).

    ``function_name`` is **required** (non-empty after strip). Use
    :data:`SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME` (``\"sine\"``) for the FittingTime
    target: ``x ~ U(-1,1)^{N×D}``, ``y = mean(sin(2π (x+1)/2)) + 0.1 ε`` (equivalently
    ``x_u ~ U(0,1)`` with ``y = mean(sin(2π x_u)) + noise``).

    Any other name builds ``f"f:{function_name}-{D}d"`` via
    :mod:`problems.pure_functions`, draws ``x ~ U(-1,1)^{N×D}``, and sets ``y`` to the
    environment reward plus ``0.1 ε``. Fitted surrogates use ``(x+1)/2`` in ``[0,1]`` (ENN and SMAC only
    when ``b_fast_only``); metrics always use the original ``y`` / ``y_test`` from the env draw.

    **Replicates:** for ``num_reps`` > 1, the full benchmark is run ``num_reps`` times with
    ``problem_seed``, ``problem_seed + 1``, … (each replicate gets a fresh synthetic
    draw). Returned :class:`BMResult` fields are the sample mean and standard error of the
    mean (finite values only; NaNs omitted from mean/SEM, or all-NaN → NaN mean and SEM).

    Reproducible RNG within one replicate: for ``\"sine\"``, ``torch.manual_seed(problem_seed + rep)``
    before the train draw and ``+1`` for the test draw inside :func:`draw_benchmark_synthetic_xy`.
    For other names, the env uses ``problem_seed + rep`` for distortion while train/test
    ``torch.rand`` use seeds ``0`` and ``1``.

    If the SMAC RF path is unavailable or raises (e.g. missing ``smac``, bad install,
    runtime fit failure), SMAC RF fields are set to ``nan``.

    **LogLik** (nats): ``sum_i log N(y_test_i | y_hat_i, v_i)`` with **predictive**
    variance aligned to noisy ``y_test`` (``0.1^2`` observation noise in the draw).
    ENN uses epistemic ``se_i^2`` plus ``0.1^2``. SMAC RF uses forest variance plus
    ``0.1^2``. DNGO uses BLR predictive variance (includes learned ``1/\\beta``).
    Exact GP and both SVGP variants use ``posterior(..., observation_noise=True)``.
    Vecchia uses ``pyvecch`` posterior variance on the original ``y`` scale.
    If ``pyvecch`` is missing or fit/predict fails, Vecchia fields are ``nan`` (same pattern as
    optional SMAC RF). For :func:`~analysis.fitting_time.fitting_time.fit_vecchia` only, set
    ``YUBO_ALLOW_PYVECCH_ON_DARWIN=0`` on macOS to return NaNs if import still crashes in your
    environment.
    """
    if num_reps < 1:
        raise ValueError("num_reps must be >= 1")

    from analysis.fitting_time.fitting_time import (
        fit_dngo,
        fit_enn,
        fit_exact_gp,
        fit_smac_rf,
        fit_svgp_default,
        fit_svgp_linear,
        fit_vecchia,
    )

    rows: list[dict[str, tuple[float, float, float]]] = []
    for rep in range(num_reps):
        seed = int(problem_seed) + rep
        x, y, x_test, y_test = draw_benchmark_synthetic_xy(N=N, D=D, function_name=function_name, problem_seed=seed)
        rows.append(
            _surrogate_metric_triples_from_tensors(
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
                b_fast_only=b_fast_only,
            )
        )
    return _aggregate_surrogate_replicates(rows)
