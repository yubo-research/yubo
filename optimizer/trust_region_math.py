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


def _mahalanobis_sq_from_factor(delta: np.ndarray, factor: np.ndarray) -> np.ndarray:
    delta_arr = np.asarray(delta, dtype=float)
    factor_arr = np.asarray(factor, dtype=float)
    if factor_arr.ndim != 2 or factor_arr.shape[0] != factor_arr.shape[1]:
        raise ValueError(f"factor must be square, got {factor_arr.shape}")
    if delta_arr.ndim != 2 or delta_arr.shape[1] != factor_arr.shape[0]:
        raise ValueError((delta_arr.shape, factor_arr.shape))
    solved = np.linalg.solve(factor_arr, delta_arr.T).T
    return np.sum(solved * solved, axis=1)


def _mahalanobis_sq_from_inv(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    delta_array = np.asarray(delta, dtype=float)
    cov_inv_array = np.asarray(cov_inv, dtype=float)
    return np.einsum("nd,de,ne->n", delta_array, cov_inv_array, delta_array)


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


def _low_rank_factor_from_cov(
    cov: np.ndarray,
    *,
    dim: int,
    lam_min: float,
    lam_max: float,
    eps: float,
    rank_cap: int | None,
) -> _LowRankFactor | None:
    try:
        eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    if eigvals.size == 0 or not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return None
    r = int(min(dim, eigvals.size))
    if rank_cap is not None:
        r = min(r, max(int(rank_cap), 0))
    if r <= 0:
        return None
    if r < eigvals.size:
        alpha0 = float(np.mean(eigvals[r:]))
    else:
        alpha0 = float(np.min(eigvals))
    alpha0 = float(np.clip(alpha0, lam_min, lam_max))
    lam = np.maximum(eigvals[:r] - alpha0, 0.0)
    total = alpha0 * float(dim) + float(np.sum(lam))
    if not np.isfinite(total) or total <= 0.0:
        return None
    scale = float(dim) / total
    alpha = alpha0 * scale
    lam = lam * scale
    return _LowRankFactor(
        sqrt_alpha=float(np.sqrt(max(alpha, 0.0))),
        basis=eigvecs[:, :r],
        sqrt_vals=np.sqrt(lam + eps),
    )


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
    num_samples = int(b.shape[0])
    num_dim = int(b.shape[1])
    if num_dim <= num_samples:
        try:
            eigvals, eigvecs = np.linalg.eigh(b.T @ b)
        except np.linalg.LinAlgError:
            return None
        order = np.argsort(eigvals)[::-1]
        lam_all = np.maximum(eigvals[order], 0.0)
        basis_all = eigvecs[:, order]
    else:
        try:
            eigvals, eigvecs = np.linalg.eigh(b @ b.T)
        except np.linalg.LinAlgError:
            return None
        order = np.argsort(eigvals)[::-1]
        lam_all = np.maximum(eigvals[order], 0.0)
        sample_basis = eigvecs[:, order]
        positive = lam_all > 1e-12
        lam_all = lam_all[positive]
        sample_basis = sample_basis[:, positive]
        if lam_all.size == 0:
            return None
        basis_all = (b.T @ sample_basis) / np.sqrt(lam_all).reshape(1, -1)
        norms = np.linalg.norm(basis_all, axis=0)
        valid = norms > 1e-12
        lam_all = lam_all[valid]
        basis_all = basis_all[:, valid] / norms[valid].reshape(1, -1)
    if lam_all.size == 0 or not np.all(np.isfinite(lam_all)) or not np.all(np.isfinite(basis_all)):
        return None
    r = int(min(dim, lam_all.size, centered.shape[0]))
    if rank_cap is not None:
        r = min(r, max(int(rank_cap), 0))
    if r <= 0:
        return None
    basis = basis_all[:, :r]
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
    return _LowRankFactor(sqrt_alpha=sqrt_alpha, basis=basis, sqrt_vals=sqrt_lam)


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
