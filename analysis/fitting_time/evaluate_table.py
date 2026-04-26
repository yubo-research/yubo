"""Console table formatting for :class:`SyntheticSineSurrogateBenchmark`."""

from __future__ import annotations

import math

from analysis.fitting_time.benchmark_table_fmt import (
    fmt_mu_loglik_sweep,
    fmt_mu_nrmse,
    fmt_ratio_vs_base,
    fmt_se,
    fmt_se_loglik_sweep,
    fmt_synthetic_time_mu,
    pm_plus_minus_column,
)

from .evaluate_metrics import SURROGATE_BENCHMARK_ROWS, BMResult


def print_synthetic_benchmark_table(results: dict[str, BMResult]) -> None:
    """Print a compact timing table: ``Fit t (s)``, ``t/t_ENN``, ``NRMSE``, ``LogLik`` as ``μ ± SE``."""
    if "enn" not in results:
        raise KeyError("print_synthetic_benchmark_table requires results['enn'] for t/t_ENN column.")

    def time_ratio_vs_enn(fit_mu: float) -> float:
        base = results["enn"].fit_seconds.mu
        if math.isnan(base) or math.isnan(fit_mu):
            return float("nan")
        if base <= 0.0:
            return 1.0 if fit_mu <= 0.0 else float("inf")
        return fit_mu / base

    rows = [results[k] for k, _ in SURROGATE_BENCHMARK_ROWS]
    fit_m = [br.fit_seconds.mu for br in rows]
    fit_s = [br.fit_seconds.se for br in rows]
    nrm_m = [br.normalized_rmse.mu for br in rows]
    nrm_s = [br.normalized_rmse.se for br in rows]
    ll_m = [br.log_likelihood.mu for br in rows]
    ll_s = [br.log_likelihood.se for br in rows]
    ratio_m = [time_ratio_vs_enn(br.fit_seconds.mu) for br in rows]

    fit_cells = pm_plus_minus_column(fit_m, fit_s, fmt_synthetic_time_mu, fmt_se)
    nrmse_cells = pm_plus_minus_column(nrm_m, nrm_s, fmt_mu_nrmse, fmt_se)
    ll_cells = pm_plus_minus_column(ll_m, ll_s, fmt_mu_loglik_sweep, fmt_se_loglik_sweep)
    ratio_cells = [fmt_ratio_vs_base(x) for x in ratio_m]

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
        body.append(
            f"{name:<{w_name}}  {fit_cells[i]:>{w_fit}}  {ratio_cells[i]:>{w_sp}}  {nrmse_cells[i]:>{w_n}}  {ll_cells[i]:>{w_ll}}",
        )
    print("\n".join([header, sep, *body]))
