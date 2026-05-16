"""Shared metrics and datatypes for surrogate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME = "sine"

SURROGATE_BENCHMARK_KEYS: tuple[str, ...] = (
    "enn",
    "enn_hnsw",
    "smac_rf",
    "dngo",
    "exact_gp",
    "svgp_default",
    "svgp_linear",
    "vecchia",
)

SURROGATE_BENCHMARK_ROWS: tuple[tuple[str, str], ...] = (
    ("enn", "ENN"),
    ("enn_hnsw", "ENN+HNSW"),
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
