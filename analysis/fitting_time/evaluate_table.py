"""Console table formatting for :class:`SyntheticSineSurrogateBenchmark`."""

from __future__ import annotations

import math

from .evaluate_metrics import SURROGATE_BENCHMARK_ROWS, BMResult


def print_synthetic_benchmark_table(results: dict[str, BMResult]) -> None:
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
