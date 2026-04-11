from __future__ import annotations

from typing import Any

import numpy as np
from enn.turbo.config.candidate_rv import CandidateRV

import optimizer.trust_region_accel as _accel
from optimizer.trust_region_math import _LowRankFactor, _normalize_weights


def _candidate_rv_name(candidate_rv: CandidateRV | None, *, default: str = "uniform") -> str:
    if candidate_rv is None:
        return default
    return str(getattr(candidate_rv, "value", candidate_rv)).lower()


def _draw_sobol_prefix(sobol_engine: Any, n: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return sobol_engine.random(0)
    n_sobol = 1 << (n - 1).bit_length()
    if hasattr(sobol_engine, "random_base2"):
        return sobol_engine.random_base2(n_sobol.bit_length() - 1)[:n]
    return sobol_engine.random(n_sobol)[:n]


def _whitened_inputs_from_sobol_samples(
    samples: np.ndarray,
    *,
    num_dim: int,
    prob: float,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    normal_pairs = (num_dim + 1) // 2
    total_cols = num_dim + 2 * normal_pairs + 1
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] < total_cols:
        raise ValueError(f"sobol_engine dimension too small for whitened sampling: need {total_cols}, got {samples.shape}")
    mask = _repair_empty_raasp_mask(samples[:, :num_dim] < prob, num_dim=num_dim, rng=rng)
    uniforms = samples[:, num_dim : num_dim + 2 * normal_pairs]
    u1 = np.clip(uniforms[:, 0::2], np.float32(1e-12), np.float32(1.0 - 1e-12))
    u2 = uniforms[:, 1::2]
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    z_tilde = np.empty((samples.shape[0], num_dim), dtype=np.float32)
    z_tilde[:, 0::2] = radius * np.cos(angle)
    if num_dim > 1:
        z_tilde[:, 1::2] = (radius * np.sin(angle))[:, : z_tilde[:, 1::2].shape[1]]
    z_tilde *= mask
    return z_tilde, samples[:, num_dim + 2 * normal_pairs]


def _sample_box_perturbations(
    lb: np.ndarray,
    ub: np.ndarray,
    num_candidates: int,
    *,
    rng: Any,
    candidate_rv: CandidateRV,
    sobol_engine: Any | None,
) -> np.ndarray:
    lb_array = np.asarray(lb, dtype=float)
    ub_array = np.asarray(ub, dtype=float)
    if candidate_rv == CandidateRV.SOBOL:
        if sobol_engine is None:
            raise ValueError("sobol_engine required for CandidateRV.SOBOL")
        samples = _draw_sobol_prefix(sobol_engine, int(num_candidates))
        return lb_array + (ub_array - lb_array) * samples
    return lb_array + (ub_array - lb_array) * rng.uniform(0.0, 1.0, size=(num_candidates, lb_array.size))


def _generate_block_raasp_candidates(
    center: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    num_candidates: int,
    *,
    rng: Any,
    candidate_rv: CandidateRV,
    sobol_engine: Any | None,
    block_slices: tuple[tuple[int, int], ...],
    block_prob: float,
) -> np.ndarray:
    x_center = np.asarray(center, dtype=float).reshape(-1)
    num_dim = int(x_center.size)
    num_blocks = int(len(block_slices))
    if num_blocks <= 0:
        raise ValueError("block_slices must be non-empty")
    prob_perturb = float(min(max(block_prob, 0.0), 1.0))
    ks = np.maximum(rng.binomial(num_blocks, prob_perturb, size=num_candidates), 1)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    for i in range(int(num_candidates)):
        block_idx = rng.choice(num_blocks, size=int(ks[i]), replace=False)
        for j in np.asarray(block_idx, dtype=np.int64):
            start, end = block_slices[int(j)]
            mask[i, int(start) : int(end)] = True
    pert = _sample_box_perturbations(
        np.asarray(lb, dtype=float),
        np.asarray(ub, dtype=float),
        int(num_candidates),
        rng=rng,
        candidate_rv=candidate_rv,
        sobol_engine=sobol_engine,
    )
    candidates = np.tile(x_center, (int(num_candidates), 1))
    if np.any(mask):
        candidates[mask] = pert[mask]
    return candidates


def _apply_block_raasp_mask(
    candidates: np.ndarray,
    *,
    rng: Any,
    candidate_rv: CandidateRV,
    sobol_engine: Any | None,
    block_slices: tuple[tuple[int, int], ...],
    block_prob: float,
) -> np.ndarray:
    x = np.asarray(candidates, dtype=float).copy()
    if x.ndim != 2:
        raise ValueError(f"candidates must be 2D, got {x.shape}")
    num_candidates, num_dim = x.shape
    num_blocks = int(len(block_slices))
    if num_blocks <= 0:
        return x
    prob_perturb = float(min(max(block_prob, 0.0), 1.0))
    ks = np.maximum(rng.binomial(num_blocks, prob_perturb, size=num_candidates), 1)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    for i in range(int(num_candidates)):
        block_idx = rng.choice(num_blocks, size=int(ks[i]), replace=False)
        for j in np.asarray(block_idx, dtype=np.int64):
            start, end = block_slices[int(j)]
            mask[i, int(start) : int(end)] = True
    pert = _sample_box_perturbations(
        np.zeros(num_dim, dtype=float),
        np.ones(num_dim, dtype=float),
        int(num_candidates),
        rng=rng,
        candidate_rv=candidate_rv,
        sobol_engine=sobol_engine,
    )
    if np.any(mask):
        x[mask] = pert[mask]
    return x


def _generate_raasp_candidates_fast_uniform(
    center: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    num_candidates: int,
    *,
    rng: Any,
    num_pert: int,
) -> np.ndarray:
    x_center = np.asarray(center, dtype=float).reshape(-1)
    lb_array = np.asarray(lb, dtype=float).reshape(-1)
    ub_array = np.asarray(ub, dtype=float).reshape(-1)
    num_dim = int(x_center.size)
    if num_dim <= 0:
        raise ValueError(num_dim)
    n = int(num_candidates)
    prob_perturb = min(float(num_pert) / float(num_dim), 1.0)
    random_values = rng.random((n, 2 * num_dim))
    mask = _repair_empty_raasp_mask(random_values[:, :num_dim] < prob_perturb, num_dim=num_dim, rng=rng)
    pert = lb_array + (ub_array - lb_array) * random_values[:, num_dim:]
    candidates = np.tile(x_center, (n, 1))
    candidates[mask] = pert[mask]
    return candidates


def _repair_empty_raasp_mask(mask: np.ndarray, *, num_dim: int, rng: Any) -> np.ndarray:
    empty = np.where(np.sum(mask, axis=1) == 0)[0]
    if empty.size > 0:
        cols = rng.integers(0, num_dim, size=empty.size)
        mask[empty, cols] = True
    return mask


def _sample_whitened_inputs(
    *,
    num_candidates: int,
    num_dim: int,
    rng: Any,
    candidate_rv: CandidateRV | None,
    prob: float,
    sobol_engine: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rv_name = _candidate_rv_name(candidate_rv)
    if rv_name == "sobol":
        if sobol_engine is None:
            raise ValueError("sobol_engine required for CandidateRV.SOBOL")
        n = int(num_candidates)
        samples = _draw_sobol_prefix(sobol_engine, n)
        return _whitened_inputs_from_sobol_samples(samples, num_dim=num_dim, prob=prob, rng=rng)
    if rv_name in {"uniform", "gpu_uniform"}:
        random_values = rng.random((num_candidates, 2 * num_dim + 1), dtype=np.float32)
        mask = _repair_empty_raasp_mask(random_values[:, :num_dim] < prob, num_dim=num_dim, rng=rng)
        z_tilde = random_values[:, num_dim : 2 * num_dim]
        z_tilde *= np.float32(2.0)
        z_tilde -= np.float32(1.0)
        z_tilde *= mask
        return z_tilde, random_values[:, -1]
    mask = _repair_empty_raasp_mask(
        rng.random((num_candidates, num_dim), dtype=np.float32) < prob,
        num_dim=num_dim,
        rng=rng,
    )
    z_tilde = rng.standard_normal((num_candidates, num_dim), dtype=np.float32)
    z_tilde *= mask
    return z_tilde, rng.random(num_candidates, dtype=np.float32)


def _whitened_sample_numpy(
    *,
    z_tilde: np.ndarray,
    u: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
    rng: Any,
) -> np.ndarray:
    z_tilde = np.asarray(z_tilde, dtype=np.float32).copy()
    norms = np.linalg.norm(z_tilde, axis=1)
    tiny = norms <= 1e-12
    if np.any(tiny):
        idx_rows = np.where(tiny)[0]
        idx_cols = rng.integers(0, num_dim, size=idx_rows.size)
        signs = np.where(rng.random(idx_rows.size) < 0.5, -1.0, 1.0)
        z_tilde[idx_rows] = 0.0
        z_tilde[idx_rows, idx_cols] = signs
        norms = np.linalg.norm(z_tilde, axis=1)
    v = z_tilde / norms.reshape(-1, 1)
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u
    else:
        rho = np.power(u, 1.0 / float(max(num_dim, 1)))
    return float(length) * rho.reshape(-1, 1) * v


def _low_rank_symmetric_sqrt_step(
    z: np.ndarray,
    low_rank: _LowRankFactor,
    *,
    use_accel: bool = False,
) -> np.ndarray:
    """Apply the symmetric square root of a low-rank covariance factor.

    For orthonormal V, sqrt(C) = sqrt(alpha) I + V diag(sqrt(alpha + lam) - sqrt(alpha)) V^T
    where C = alpha I + V diag(lam) V^T.
    """
    z_arr = np.asarray(z, dtype=float)
    sqrt_alpha = float(low_rank.sqrt_alpha)
    basis = np.asarray(low_rank.basis, dtype=float)
    sqrt_vals = np.asarray(low_rank.sqrt_vals, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != z_arr.shape[1]:
        raise ValueError(f"basis shape {basis.shape} incompatible with z shape {z_arr.shape}")
    if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
        raise ValueError((basis.shape, sqrt_vals.shape))
    if basis.shape[1] == 0:
        return sqrt_alpha * z_arr
    alpha = sqrt_alpha * sqrt_alpha
    lam = np.square(sqrt_vals)
    gains = np.sqrt(alpha + lam) - sqrt_alpha
    proj = _accel.matmul(z_arr, basis) if use_accel else z_arr @ basis
    coeff = proj * gains.reshape(1, -1)
    correction = _accel.low_rank_step(coeff, basis) if use_accel else coeff @ basis.T
    return sqrt_alpha * z_arr + correction


def _low_rank_mahalanobis_sq(
    delta: np.ndarray,
    low_rank: _LowRankFactor,
    *,
    use_accel: bool = False,
) -> np.ndarray:
    """Compute delta^T C^{-1} delta for C = alpha I + V diag(lam) V^T via Woodbury."""
    delta_arr = np.asarray(delta, dtype=float)
    sqrt_alpha = float(low_rank.sqrt_alpha)
    alpha = max(sqrt_alpha * sqrt_alpha, 1e-12)
    inv_alpha = 1.0 / alpha
    basis = np.asarray(low_rank.basis, dtype=float)
    sqrt_vals = np.asarray(low_rank.sqrt_vals, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != delta_arr.shape[1]:
        raise ValueError(f"basis shape {basis.shape} incompatible with delta shape {delta_arr.shape}")
    if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
        raise ValueError((basis.shape, sqrt_vals.shape))
    if basis.shape[1] == 0:
        return inv_alpha * np.sum(delta_arr * delta_arr, axis=1)
    lam = np.square(sqrt_vals)
    beta = inv_alpha * lam / (alpha + lam)
    if use_accel:
        return _accel.low_rank_metric(delta_arr, basis, beta, inv_alpha)
    proj = delta_arr @ basis
    return inv_alpha * np.sum(delta_arr * delta_arr, axis=1) - np.sum(proj * proj * beta.reshape(1, -1), axis=1)


def _full_factor_from_direction(
    direction: np.ndarray,
    *,
    dim: int,
    lam_min: float,
    eps: float,
) -> np.ndarray:
    unit = np.asarray(direction, dtype=float).reshape(-1)
    if unit.shape != (dim,):
        raise ValueError((unit.shape, dim))
    e1 = np.zeros(dim, dtype=float)
    e1[0] = 1.0
    diff = e1 - unit
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm <= 1e-12:
        basis = np.eye(dim, dtype=float)
    else:
        v = diff / diff_norm
        basis = np.eye(dim, dtype=float) - 2.0 * np.outer(v, v)
    lam_large = float(dim)
    lam_small = float(lam_min)
    total = lam_large + (dim - 1) * lam_small
    scale = float(dim) / total
    eigvals = np.full(dim, lam_small * scale, dtype=float)
    eigvals[0] = lam_large * scale
    return basis * np.sqrt(eigvals + eps).reshape(1, -1)


def _low_rank_factor_from_direction(
    direction: np.ndarray,
    *,
    dim: int,
    eps: float,
) -> _LowRankFactor:
    unit = np.asarray(direction, dtype=float).reshape(-1)
    if unit.shape != (dim,):
        raise ValueError((unit.shape, dim))
    alpha0 = 1e-4
    scale = 1.0 / (1.0 + alpha0)
    alpha = alpha0 * scale
    lam = float(dim) * scale
    return _LowRankFactor(
        sqrt_alpha=float(np.sqrt(alpha)),
        basis=unit.reshape(-1, 1),
        sqrt_vals=np.array([np.sqrt(lam + eps)], dtype=float),
    )


def _prepare_gradient_geometry_inputs(
    *,
    delta_x: np.ndarray | Any,
    delta_y: np.ndarray | Any,
    weights: np.ndarray | Any,
    num_dim: int,
    eps_norm: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    dx = np.asarray(delta_x, dtype=float)
    if dx.ndim != 2 or dx.shape[0] == 0:
        return None
    if dx.shape[1] != num_dim:
        raise ValueError(f"delta_x has incompatible shape {dx.shape} for num_dim={num_dim}")
    dy = np.asarray(delta_y, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if dy.shape[0] != dx.shape[0] or w.shape[0] != dx.shape[0]:
        raise ValueError((dy.shape, w.shape, dx.shape))
    w = _normalize_weights(w)
    if w is None:
        return None
    norms = np.linalg.norm(dx, axis=1)
    scale = np.abs(dy) / np.maximum(norms, float(eps_norm))
    scale = np.where(np.isfinite(scale), scale, 0.0)
    if not np.any(scale > 0.0):
        return None
    return dx * scale.reshape(-1, 1), w


def _generate_raasp_mask_fast(
    *,
    num_dim: int,
    num_candidates: int,
    rng: Any,
    num_pert: int,
) -> np.ndarray:
    prob_perturb = min(float(num_pert) / float(num_dim), 1.0)
    mask = rng.random((int(num_candidates), num_dim)) < prob_perturb
    return _repair_empty_raasp_mask(mask, num_dim=num_dim, rng=rng)


def _generate_raasp_candidates_fast_sobol(
    center: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    num_candidates: int,
    *,
    rng: Any,
    sobol_engine: Any,
    num_pert: int,
) -> np.ndarray:
    x_center = np.asarray(center, dtype=float).reshape(-1)
    lb_array = np.asarray(lb, dtype=float).reshape(-1)
    ub_array = np.asarray(ub, dtype=float).reshape(-1)
    num_dim = int(x_center.size)
    if num_dim <= 0:
        raise ValueError(num_dim)
    n = int(num_candidates)
    mask = _generate_raasp_mask_fast(num_dim=num_dim, num_candidates=n, rng=rng, num_pert=num_pert)
    sobol_samples = _draw_sobol_prefix(sobol_engine, n)
    pert = lb_array + (ub_array - lb_array) * sobol_samples
    candidates = np.tile(x_center, (n, 1))
    candidates[mask] = pert[mask]
    return candidates
