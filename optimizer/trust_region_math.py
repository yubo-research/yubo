"""Low-level math helpers for trust region geometry. Split from trust_region_utils to reduce statements."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class _LowRankFactor(NamedTuple):
    sqrt_alpha: float
    basis: np.ndarray
    sqrt_vals: np.ndarray


def _normalize_weights(weights: np.ndarray) -> np.ndarray | None:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return w / total


def _trace_normalize(cov: np.ndarray, dim: int) -> np.ndarray:
    trace = float(np.trace(cov))
    if not np.isfinite(trace) or trace <= 0.0:
        return np.eye(dim, dtype=float)
    return cov / trace * float(dim)


def _ensure_spd(
    cov: np.ndarray,
    *,
    jitter: float = 1e-8,
    max_tries: int = 6,
) -> np.ndarray:
    mat = np.asarray(cov, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"cov must be square, got {mat.shape}")
    mat = 0.5 * (mat + mat.T)
    eye = np.eye(mat.shape[0], dtype=float)
    scale = float(max(1.0, np.max(np.abs(mat))))
    jitter_val = float(max(jitter, 1e-12))
    for _ in range(max_tries):
        trial = mat + (jitter_val * scale) * eye
        try:
            np.linalg.cholesky(trial)
            return trial
        except np.linalg.LinAlgError:
            jitter_val *= 10.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, jitter_val * scale)
    return (eigvecs * eigvals.reshape(1, -1)) @ eigvecs.T


def _clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    num_candidates, num_dim = step.shape
    t = np.ones((num_candidates,), dtype=float)
    for j in range(num_dim):
        sj = step[:, j]
        if float(x_center[j]) < 0.0 or float(x_center[j]) > 1.0:
            raise ValueError("x_center must lie in [0, 1]^D")
        pos = sj > 0.0
        if np.any(pos):
            t[pos] = np.minimum(t[pos], (1.0 - x_center[j]) / sj[pos])
        neg = sj < 0.0
        if np.any(neg):
            t[neg] = np.minimum(t[neg], (0.0 - x_center[j]) / sj[neg])
    t = np.clip(t, 0.0, 1.0)
    return x_center.reshape(1, -1) + step * t.reshape(-1, 1)


def _ray_scale_to_unit_box(x_center: np.ndarray, x: np.ndarray) -> np.ndarray:
    center = np.asarray(x_center, dtype=float)
    return _clip_to_unit_box(center, np.asarray(x, dtype=float) - center)


def _mahalanobis_sq(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    solved = np.linalg.solve(cov, delta.T).T
    return np.sum(delta * solved, axis=1)


def _clip_and_rescale_eigs(
    eigvals: np.ndarray,
    *,
    dim: int,
    lam_min: float,
    lam_max: float,
) -> np.ndarray | None:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.clip(eigvals, lam_min, lam_max)
    total = float(np.sum(eigvals))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return eigvals * (float(dim) / total)


def _full_factor(
    cov: np.ndarray,
    *,
    dim: int,
    lam_min: float,
    lam_max: float,
    eps: float,
) -> np.ndarray | None:
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    eigvals = _clip_and_rescale_eigs(
        eigvals,
        dim=dim,
        lam_min=lam_min,
        lam_max=lam_max,
    )
    if eigvals is None:
        return None
    return eigvecs * np.sqrt(eigvals + eps).reshape(1, -1)


def _low_rank_factor(
    centered: np.ndarray,
    weights: np.ndarray,
    *,
    dim: int,
    lam_min: float,
    lam_max: float,
    eps: float,
    rank_cap: int | None,
) -> _LowRankFactor | None:
    b = centered * np.sqrt(weights).reshape(-1, 1)
    try:
        _u, svals, vt = np.linalg.svd(b, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if svals.size == 0 or not np.all(np.isfinite(svals)):
        return None
    lam_all = np.square(svals)
    if not np.all(np.isfinite(lam_all)):
        return None
    r = int(min(dim, lam_all.size, centered.shape[0]))
    if rank_cap is not None:
        r = min(r, max(int(rank_cap), 0))
    if r <= 0:
        return None
    v = vt[:r].T
    lam = lam_all[:r]
    total = float(np.sum(lam))
    if not np.isfinite(total) or total <= 0.0:
        return None
    lam = lam / total * float(dim)
    lam = np.clip(lam, lam_min, lam_max)
    lam = _clip_and_rescale_eigs(lam, dim=dim, lam_min=lam_min, lam_max=lam_max)
    if lam is None:
        return None
    alpha0 = 1e-4
    trace_total = alpha0 * float(dim) + float(np.sum(lam))
    scale = float(dim) / trace_total
    alpha = alpha0 * scale
    lam = lam * scale
    sqrt_alpha = float(np.sqrt(alpha))
    sqrt_lam = np.sqrt(lam + eps)
    return _LowRankFactor(sqrt_alpha=sqrt_alpha, basis=v, sqrt_vals=sqrt_lam)


def _add_sparse_axis(step: np.ndarray, z: np.ndarray, scale: float) -> None:
    if scale == 0.0:
        return
    nz_cols = np.where(np.any(z != 0.0, axis=0))[0]
    for j in nz_cols:
        rows = np.where(z[:, j] != 0.0)[0]
        if rows.size == 0:
            continue
        step[rows, j] += scale * z[rows, j]


def _apply_full_factor(z: np.ndarray, factor: np.ndarray) -> np.ndarray:
    num_candidates, num_dim = z.shape
    if factor.shape != (num_dim, num_dim):
        raise ValueError(f"full factor must be ({num_dim}, {num_dim}), got {factor.shape}")
    step = np.zeros((num_candidates, num_dim), dtype=float)
    nz_cols = np.where(np.any(z != 0.0, axis=0))[0]
    for j in nz_cols:
        rows = np.where(z[:, j] != 0.0)[0]
        if rows.size == 0:
            continue
        step[rows] += z[rows, j].reshape(-1, 1) * factor[:, j].reshape(1, -1)
    return step
