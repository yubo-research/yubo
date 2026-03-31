"""ENN hyperparameter sweeps on synthetic benchmark targets."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

from analysis.fitting_time.evaluate import (
    _mean_and_sem,
    draw_benchmark_synthetic_xy,
    env_action_coords_to_surrogate_unit_x,
    normalize_benchmark_function_name,
    normalized_rmse,
    predictive_gaussian_log_likelihood,
)
from analysis.fitting_time.fitting_time import fit_enn

__all__ = ["sweep"]


def _nonfinite_token(x: float) -> str | None:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return None


def _fmt_mu_nrmse_sweep(x: float) -> str:
    t = _nonfinite_token(x)
    if t is not None:
        return t
    return f"{x:.6g}"


def _fmt_se_nrmse_sweep(x: float) -> str:
    t = _nonfinite_token(x)
    if t is not None:
        return t
    return f"{x:.2g}"


def _fmt_mu_loglik_sweep(x: float) -> str:
    t = _nonfinite_token(x)
    if t is not None:
        return t
    if abs(x) >= 10.0:
        return f"{x:.2f}"
    return f"{x:.6g}"


def _fmt_se_loglik_sweep(x: float) -> str:
    t = _nonfinite_token(x)
    if t is not None:
        return t
    if abs(x) >= 1.0:
        return f"{x:.0f}"
    return f"{x:.2g}"


def _pm_cells_sweep(
    mus: list[float],
    ses: list[float],
    fmt_mu: Callable[[float], str],
    fmt_se_fn: Callable[[float], str],
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


def _nrmse_ll_lists_for_kp(
    *,
    N: int,
    D: int,
    function_name: str,
    k_i: int,
    p_i: int,
    num_reps: int,
    problem_seed: int,
    kp_index: int,
    num_fit_candidates: int,
) -> tuple[list[float], list[float]]:
    nrs: list[float] = []
    lls: list[float] = []
    for r in range(num_reps):
        seed = int(problem_seed) + kp_index * 10_007 + r * 1_000_003
        x, y, x_test, y_test = draw_benchmark_synthetic_xy(N=N, D=D, function_name=function_name, problem_seed=seed)
        x_s = env_action_coords_to_surrogate_unit_x(x)
        x_ts = env_action_coords_to_surrogate_unit_x(x_test)
        train_x = x_s.detach().cpu().numpy().astype(np.float64)
        train_y = y.detach().cpu().numpy().astype(np.float64)
        x_test_np = x_ts.detach().cpu().numpy().astype(np.float64)
        y_test = y_test.to(dtype=torch.float64) if isinstance(y_test, torch.Tensor) else y_test
        _, y_hat, pred_var = fit_enn(
            train_x,
            train_y,
            x_test_np,
            k=k_i,
            num_fit_samples=min(p_i, N),
            num_fit_candidates=num_fit_candidates,
            rng=np.random.default_rng(seed + 1),
        )
        nrs.append(normalized_rmse(y_test, y_hat))
        lls.append(predictive_gaussian_log_likelihood(y_test, y_hat, pred_var))
    return nrs, lls


def sweep(
    *,
    N: int,
    D: int,
    function_name: str,
    kp_pairs: Sequence[tuple[int, int]],
    problem_seed: int = 0,
    num_reps: int = 10,
    num_fit_candidates: int = 100,
) -> None:
    """Fit ENN ``num_reps`` times per ``(K, P)`` pair and print NRMSE / LogLik as mean ± SEM.

    Each replicate draws ``N`` train and test points via :func:`draw_benchmark_synthetic_xy`
    (same protocol as the timing benchmark). ``K`` is passed to :func:`enn_fit` as ``k``;
    ``P`` is ``num_fit_samples`` (clamped to ``[1, N]``). Metrics match
    :func:`benchmark_synthetic_sine_surrogates` (noisy ``y_test``, predictive variance with
    ``0.1^2`` observation noise).

    SEM is sample standard deviation / ``sqrt(num_reps)`` with ``ddof=1`` (``0`` when
    ``num_reps == 1``), matching :class:`~analysis.fitting_time.evaluate.MuSe`. Prints::

        K  P  NRMSE           LogLik
    """
    if N < 1 or D < 1:
        raise ValueError("N and D must be positive")
    if num_reps < 1:
        raise ValueError("num_reps must be >= 1")
    if not kp_pairs:
        raise ValueError("kp_pairs must be non-empty")
    fn = normalize_benchmark_function_name(function_name)

    wk, wp = 6, 8
    rows_kp: list[tuple[int, int]] = []
    nrm_mus: list[float] = []
    nrm_ses: list[float] = []
    ll_mus: list[float] = []
    ll_ses: list[float] = []

    for idx, (k_raw, p_raw) in enumerate(kp_pairs):
        k_i = int(k_raw)
        p_i = int(p_raw)
        if k_i < 1 or p_i < 1:
            raise ValueError(f"K and P must be >= 1, got K={k_i}, P={p_i}")
        nrs, lls = _nrmse_ll_lists_for_kp(
            N=N,
            D=D,
            function_name=fn,
            k_i=k_i,
            p_i=p_i,
            num_reps=num_reps,
            problem_seed=problem_seed,
            kp_index=idx,
            num_fit_candidates=num_fit_candidates,
        )
        ms_n = _mean_and_sem(nrs)
        ms_l = _mean_and_sem(lls)
        rows_kp.append((k_i, p_i))
        nrm_mus.append(ms_n.mu)
        nrm_ses.append(ms_n.se)
        ll_mus.append(ms_l.mu)
        ll_ses.append(ms_l.se)

    nrm_cells = _pm_cells_sweep(nrm_mus, nrm_ses, _fmt_mu_nrmse_sweep, _fmt_se_nrmse_sweep)
    ll_cells = _pm_cells_sweep(ll_mus, ll_ses, _fmt_mu_loglik_sweep, _fmt_se_loglik_sweep)
    h_n, h_l = "NRMSE", "LogLik"
    wn = max(len(h_n), max(len(c) for c in nrm_cells))
    wl = max(len(h_l), max(len(c) for c in ll_cells))
    header = f"{'K':>{wk}}  {'P':>{wp}}  {h_n:>{wn}}  {h_l:>{wl}}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for (k_i, p_i), cn, cl in zip(rows_kp, nrm_cells, ll_cells, strict=True):
        print(f"{k_i:>{wk}}  {p_i:>{wp}}  {cn:>{wn}}  {cl:>{wl}}")
