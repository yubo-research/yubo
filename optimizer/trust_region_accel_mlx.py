"""MLX backend helpers for trust-region accelerated kernels."""

from __future__ import annotations

import numpy as np

from optimizer.trust_region_ops import (
    BackendOps,
)
from optimizer.trust_region_ops import (
    clip_step_formula as _clip_step_formula,
)
from optimizer.trust_region_ops import (
    fused_ellipsoid_generate_formula as _fused_ellipsoid_generate_formula,
)
from optimizer.trust_region_ops import (
    fused_low_rank_candidates_formula as _fused_low_rank_candidates_formula,
)
from optimizer.trust_region_ops import (
    fused_metric_candidates_formula as _fused_metric_candidates_formula,
)
from optimizer.trust_region_ops import (
    fused_whitened_ellipsoid_candidates_formula as _fused_whitened_ellipsoid_candidates_formula,
)
from optimizer.trust_region_ops import (
    low_rank_metric_formula as _low_rank_metric_formula,
)
from optimizer.trust_region_ops import (
    low_rank_step_formula as _low_rank_step_formula,
)
from optimizer.trust_region_ops import (
    low_rank_step_with_sparse_formula as _low_rank_step_with_sparse_formula,
)
from optimizer.trust_region_ops import (
    mahalanobis_from_cov_formula as _mahalanobis_from_cov_formula,
)
from optimizer.trust_region_ops import (
    mahalanobis_from_factor_formula as _mahalanobis_from_factor_formula,
)
from optimizer.trust_region_ops import (
    mahalanobis_sq_formula as _mahalanobis_sq_formula,
)
from optimizer.trust_region_ops import (
    whitened_sample_formula as _whitened_sample_formula,
)

mx_module = None
ellipsoid_needs_inverse = False


def available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_loaded():
    global mx_module
    if mx_module is not None:
        return
    import mlx.core as loaded_mx

    mx_module = loaded_mx


def mlx_module():
    ensure_loaded()
    return mx_module


def to_mlx(a: np.ndarray):
    return mlx_module().array(np.asarray(a, dtype=np.float32))


def from_mlx(a) -> np.ndarray:
    return np.asarray(a)


def _ops() -> BackendOps:
    mx = mlx_module()
    return BackendOps(
        sum_rows=lambda x: mx.sum(x, axis=1),
        min_rows=lambda x: mx.min(x, axis=1),
        norm_rows=lambda x: mx.linalg.norm(x, axis=1),
        where=mx.where,
        power=mx.power,
        sqrt=mx.sqrt,
        matmul=lambda a, b: a @ b,
        solve=lambda a, b: mx.linalg.solve(a, b, stream=mx.cpu),
        solve_triangular=lambda a, b: mx.linalg.solve(a, b, stream=mx.cpu),
    )


def clip_step(x_center, step):
    return _clip_step_formula(x_center, step, _ops())


def factorize_cov(cov: np.ndarray, *, need_inv: bool = True):
    mx = mlx_module()
    cov_mx = to_mlx(cov)
    try:
        chol = mx.linalg.cholesky(cov_mx)
    except ValueError:
        chol = mx.linalg.cholesky(cov_mx, stream=mx.cpu)
    inv = None
    if need_inv:
        inv = mx.linalg.inv(cov_mx, stream=mx.cpu)
        mx.eval(chol, inv)
    else:
        mx.eval(chol)
    return chol, inv


def mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    out = _mahalanobis_sq_formula(to_mlx(delta), to_mlx(cov_inv), _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def mahalanobis_sq_from_factor(delta: np.ndarray, chol: np.ndarray) -> np.ndarray:
    out = _mahalanobis_from_factor_formula(to_mlx(delta), to_mlx(chol), _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    out = _mahalanobis_from_cov_formula(to_mlx(delta), to_mlx(cov), _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    out = _low_rank_step_formula(to_mlx(coeff), to_mlx(basis), _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def low_rank_step_with_sparse(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> np.ndarray:
    out = _low_rank_step_with_sparse_formula(to_mlx(coeff), to_mlx(basis), to_mlx(z), scale, _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    out = _low_rank_metric_formula(to_mlx(delta), to_mlx(basis), to_mlx(beta), inv_alpha, _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    mx = mlx_module()
    out = clip_step(to_mlx(x_center).reshape(1, -1), to_mlx(step))
    mx.eval(out)
    return from_mlx(out)


def cholesky(cov: np.ndarray) -> np.ndarray:
    chol, _ = factorize_cov(cov)
    return from_mlx(chol)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mx = mlx_module()
    out = to_mlx(a) @ to_mlx(b)
    mx.eval(out)
    return from_mlx(out)


def whitened_sample(
    z_tilde: np.ndarray,
    u: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
) -> np.ndarray:
    out = _whitened_sample_formula(to_mlx(z_tilde), to_mlx(u), length, radial_mode, num_dim, _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def fused_sobol_ellipsoid_candidates(
    sobol_samples: np.ndarray,
    x_center: np.ndarray,
    chol,
    length: float,
    radial_mode: str,
    num_dim: int,
    prob: float,
    radius2: float,
) -> np.ndarray:
    mx = mlx_module()
    chol_mx = to_mlx(chol)
    samples = to_mlx(sobol_samples)
    normal_pairs = (num_dim + 1) // 2
    uniforms = samples[:, num_dim : num_dim + 2 * normal_pairs]
    u1 = mx.clip(uniforms[:, 0::2], np.float32(1e-12), np.float32(1.0 - 1e-12))
    u2 = uniforms[:, 1::2]
    radius = mx.sqrt(-2.0 * mx.log(u1))
    angle = 2.0 * np.pi * u2
    z_pairs = mx.stack((radius * mx.cos(angle), radius * mx.sin(angle)), axis=2)
    z_tilde = z_pairs.reshape(samples.shape[0], 2 * normal_pairs)[:, :num_dim]
    z_tilde = mx.where(samples[:, :num_dim] < np.float32(prob), z_tilde, np.float32(0.0))
    u = samples[:, num_dim + 2 * normal_pairs]
    z = _whitened_sample_formula(z_tilde, u, length, radial_mode, num_dim, _ops())
    out = _fused_ellipsoid_generate_formula(z, to_mlx(x_center).reshape(1, -1), chol_mx, radius2, _ops())
    mx.eval(out)
    return from_mlx(out)


def fused_whitened_ellipsoid_candidates(
    z_tilde: np.ndarray,
    u: np.ndarray,
    x_center: np.ndarray,
    chol,
    length: float,
    radial_mode: str,
    num_dim: int,
    radius2: float,
) -> np.ndarray:
    out = _fused_whitened_ellipsoid_candidates_formula(
        to_mlx(z_tilde),
        to_mlx(u),
        to_mlx(x_center).reshape(1, -1),
        to_mlx(chol),
        length,
        radial_mode,
        num_dim,
        radius2,
        _ops(),
    )
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def fused_metric_candidates(
    z: np.ndarray,
    x_center: np.ndarray,
    cov_factor: np.ndarray,
    length: float,
) -> np.ndarray:
    out = _fused_metric_candidates_formula(to_mlx(z), to_mlx(x_center).reshape(1, -1), to_mlx(cov_factor).T, length, _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def fused_low_rank_candidates(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    sparse_scale: float,
    x_center: np.ndarray,
    length: float,
) -> np.ndarray:
    out = _fused_low_rank_candidates_formula(
        to_mlx(coeff),
        to_mlx(basis),
        to_mlx(z),
        sparse_scale,
        to_mlx(x_center).reshape(1, -1),
        length,
        _ops(),
    )
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)


def fused_ellipsoid_generate(
    z: np.ndarray,
    x_center: np.ndarray,
    chol,
    radius2: float,
) -> np.ndarray:
    out = _fused_ellipsoid_generate_formula(to_mlx(z), to_mlx(x_center).reshape(1, -1), to_mlx(chol), radius2, _ops())
    mx = mlx_module()
    mx.eval(out)
    return from_mlx(out)
