from __future__ import annotations

import math
from collections.abc import Callable


__all__ = [
    "fmt_mu_nrmse",
    "fmt_mu_loglik_sweep",
    "fmt_nonfinite",
    "fmt_ratio_vs_base",
    "fmt_se",
    "fmt_se_loglik_sweep",
    "fmt_se_nrmse_sweep",
    "fmt_synthetic_time_mu",
    "pm_plus_minus_column",
]


def fmt_nonfinite(x: float) -> str | None:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return None


def _fmt_finite_spec(x: float, spec: str) -> str:
    t = fmt_nonfinite(x)
    if t is not None:
        return t
    return format(x, spec)


def fmt_synthetic_time_mu(x: float) -> str:
    t = fmt_nonfinite(x)
    if t is not None:
        return t
    if x == 0.0:
        return "0"
    if abs(x) < 0.1:
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-") else "0"
    return f"{x:.6g}"


def fmt_mu_nrmse(x: float) -> str:
    return _fmt_finite_spec(x, ".6g")


def fmt_se(x: float) -> str:
    return _fmt_finite_spec(x, ".2g")


def fmt_se_nrmse_sweep(x: float) -> str:
    return fmt_se(x)


def _fmt_loglik_sweep(x: float, *, large_abs: float, large_spec: str, small_spec: str) -> str:
    t = fmt_nonfinite(x)
    if t is not None:
        return t
    if abs(x) >= large_abs:
        return format(x, large_spec)
    return format(x, small_spec)


def fmt_mu_loglik_sweep(x: float) -> str:
    return _fmt_loglik_sweep(x, large_abs=10.0, large_spec=".2f", small_spec=".6g")


def fmt_se_loglik_sweep(x: float) -> str:
    return _fmt_loglik_sweep(x, large_abs=1.0, large_spec=".0f", small_spec=".2g")


def fmt_ratio_vs_base(x: float) -> str:
    t = fmt_nonfinite(x)
    if t is not None:
        return t
    r = round(x)
    if abs(x - r) < 1e-5:
        return str(int(r))
    return f"{x:.1f}"


def pm_plus_minus_column(
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
