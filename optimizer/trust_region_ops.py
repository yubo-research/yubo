from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BackendOps:
    sum_rows: Callable[[Any], Any]
    min_rows: Callable[[Any], Any]
    norm_rows: Callable[[Any], Any]
    where: Callable[[Any, Any, Any], Any]
    power: Callable[[Any, Any], Any]
    sqrt: Callable[[Any], Any]
    matmul: Callable[[Any, Any], Any]
    solve: Callable[[Any, Any], Any]
    solve_triangular: Callable[[Any, Any], Any]


def mahalanobis_sq_formula(delta: Any, cov_inv: Any, ops: BackendOps) -> Any:
    return ops.sum_rows(delta * ops.matmul(delta, cov_inv))


def mahalanobis_from_cov_formula(delta: Any, cov: Any, ops: BackendOps) -> Any:
    solved = ops.solve(cov, delta.T).T
    return ops.sum_rows(delta * solved)


def mahalanobis_from_factor_formula(delta: Any, chol: Any, ops: BackendOps) -> Any:
    solved = ops.solve_triangular(chol, delta.T).T
    return ops.sum_rows(solved * solved)


def low_rank_step_formula(coeff: Any, basis: Any, ops: BackendOps) -> Any:
    return ops.matmul(coeff, basis.T)


def low_rank_step_with_sparse_formula(coeff: Any, basis: Any, z: Any, scale: float, ops: BackendOps) -> Any:
    step = low_rank_step_formula(coeff, basis, ops)
    if scale != 0.0:
        step = step + float(scale) * z
    return step


def low_rank_metric_formula(delta: Any, basis: Any, beta: Any, inv_alpha: float, ops: BackendOps) -> Any:
    proj = ops.matmul(delta, basis)
    term1 = inv_alpha * ops.sum_rows(delta * delta)
    term2 = ops.sum_rows(proj * proj * beta)
    return term1 - term2


def clip_step_formula(x_center: Any, step: Any, ops: BackendOps) -> Any:
    safe_step = ops.where(step == 0.0, 1.0, step)
    t_pos = (1.0 - x_center) / safe_step
    t_neg = -x_center / safe_step
    t = ops.where(step > 0.0, t_pos, ops.where(step < 0.0, t_neg, 1e30))
    t_min_raw = ops.min_rows(t)
    t_min = ops.where(t_min_raw < 0.0, 0.0, t_min_raw)
    t_min = ops.where(t_min > 1.0, 1.0, t_min)
    return x_center + step * t_min[:, None]


def whitened_sample_formula(
    z_tilde: Any,
    u: Any,
    length: float,
    radial_mode: str,
    num_dim: int,
    ops: BackendOps,
) -> Any:
    norms = ops.norm_rows(z_tilde)
    safe_norms = ops.where(norms > 1e-12, norms, 1.0)
    directions = z_tilde / safe_norms[:, None]
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u
    else:
        rho = ops.power(u, 1.0 / max(num_dim, 1))
    return float(length) * rho[:, None] * directions


def fused_metric_candidates_formula(z: Any, x_center: Any, cov_factor_t: Any, length: float, ops: BackendOps) -> Any:
    step = ops.matmul(z, cov_factor_t) * float(length)
    return clip_step_formula(x_center, step, ops)


def fused_low_rank_candidates_formula(
    coeff: Any,
    basis: Any,
    z: Any,
    sparse_scale: float,
    x_center: Any,
    length: float,
    ops: BackendOps,
) -> Any:
    step = low_rank_step_with_sparse_formula(coeff, basis, z, sparse_scale, ops) * float(length)
    return clip_step_formula(x_center, step, ops)


def fused_ellipsoid_generate_formula(z: Any, x_center: Any, chol: Any, radius2: float, ops: BackendOps) -> Any:
    step = ops.matmul(z, chol.T)
    candidates = clip_step_formula(x_center, step, ops)
    delta = candidates - x_center
    dist2 = mahalanobis_from_factor_formula(delta, chol, ops)
    dist2_safe = ops.where(dist2 > 1e-12, dist2, 1e-12)
    scale = ops.where(
        dist2 > float(radius2) * (1.0 + 1e-8),
        ops.sqrt(float(radius2) / dist2_safe),
        1.0,
    )
    return clip_step_formula(x_center, delta * scale[:, None], ops)


def fused_whitened_ellipsoid_candidates_formula(
    z_tilde: Any,
    u: Any,
    x_center: Any,
    chol: Any,
    length: float,
    radial_mode: str,
    num_dim: int,
    radius2: float,
    ops: BackendOps,
) -> Any:
    z = whitened_sample_formula(z_tilde, u, length, radial_mode, num_dim, ops)
    return fused_ellipsoid_generate_formula(z, x_center, chol, radius2, ops)
