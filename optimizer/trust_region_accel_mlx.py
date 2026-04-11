"""MLX backend helpers for trust-region accelerated kernels."""

from __future__ import annotations

import numpy as np

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


def clip_step(x_center, step):
    mx = mlx_module()
    zero = np.float32(0.0)
    one = np.float32(1.0)
    inf = np.float32(1e30)
    abs_step = mx.abs(step)
    safe_abs_step = mx.where(abs_step > zero, abs_step, one)
    inv_abs_step = one / safe_abs_step
    upper = (one - x_center) * inv_abs_step
    lower = x_center * inv_abs_step
    limits = mx.where(step > zero, upper, mx.where(step < zero, lower, inf))
    t_min = mx.minimum(mx.maximum(mx.min(limits, axis=1), zero), one)
    return x_center + step * t_min[:, None]


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
    mx = mlx_module()
    delta_mx = to_mlx(delta)
    cov_inv_mx = to_mlx(cov_inv)
    out = mx.sum(delta_mx * (delta_mx @ cov_inv_mx), axis=1)
    mx.eval(out)
    return from_mlx(out)


def mahalanobis_sq_from_factor(delta: np.ndarray, chol: np.ndarray) -> np.ndarray:
    mx = mlx_module()
    delta_mx = to_mlx(delta)
    chol_mx = to_mlx(chol)
    solved = mx.linalg.solve(chol_mx, delta_mx.T, stream=mx.cpu).T
    out = mx.sum(solved * solved, axis=1)
    mx.eval(out)
    return from_mlx(out)


def mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    chol, _ = factorize_cov(cov, need_inv=False)
    return mahalanobis_sq_from_factor(delta, chol)


def low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    mx = mlx_module()
    out = to_mlx(coeff) @ to_mlx(basis).T
    mx.eval(out)
    return from_mlx(out)


def low_rank_step_with_sparse(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> np.ndarray:
    mx = mlx_module()
    out = to_mlx(coeff) @ to_mlx(basis).T
    if scale != 0.0:
        out = out + np.float32(scale) * to_mlx(z)
    mx.eval(out)
    return from_mlx(out)


def low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    mx = mlx_module()
    delta_mx = to_mlx(delta)
    basis_mx = to_mlx(basis)
    beta_mx = to_mlx(beta)
    proj = delta_mx @ basis_mx
    term1 = mx.array(np.float32(inv_alpha)) * mx.sum(delta_mx * delta_mx, axis=1)
    term2 = mx.sum(proj * proj * beta_mx, axis=1)
    out = term1 - term2
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
    mx = mlx_module()
    z = to_mlx(z_tilde)
    norms = mx.linalg.norm(z, axis=1)
    safe_norms = mx.where(norms > 1e-12, norms, mx.array(1.0))
    directions = z / safe_norms[:, None]
    u_mx = to_mlx(u)
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u_mx
    else:
        rho = mx.power(u_mx, 1.0 / max(num_dim, 1))
    out = np.float32(length) * rho[:, None] * directions
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
    norms = mx.linalg.norm(z_tilde, axis=1)
    safe_norms = mx.where(norms > 1e-12, norms, mx.array(1.0))
    directions = z_tilde / safe_norms[:, None]
    u = samples[:, num_dim + 2 * normal_pairs]
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u
    else:
        rho = mx.power(u, 1.0 / max(num_dim, 1))
    center = to_mlx(x_center).reshape(1, -1)
    step = (directions @ chol_mx.T) * (np.float32(length) * rho)[:, None]
    candidates = clip_step(center, step)
    delta = candidates - center
    solved = mx.linalg.solve(chol_mx, delta.T, stream=mx.cpu).T
    dist2 = mx.sum(solved * solved, axis=1)
    scale = mx.where(
        dist2 > np.float32(radius2) * (1.0 + 1e-8),
        mx.sqrt(np.float32(radius2) / mx.maximum(dist2, np.float32(1e-12))),
        mx.ones_like(dist2),
    )
    out = clip_step(center, delta * scale[:, None])
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
    mx = mlx_module()
    chol_mx = to_mlx(chol)
    z = to_mlx(z_tilde)
    norms = mx.linalg.norm(z, axis=1)
    safe_norms = mx.where(norms > 1e-12, norms, mx.array(1.0))
    directions = z / safe_norms[:, None]
    u_mx = to_mlx(u)
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u_mx
    else:
        rho = mx.power(u_mx, 1.0 / max(num_dim, 1))
    center = to_mlx(x_center).reshape(1, -1)
    step = (directions @ chol_mx.T) * (np.float32(length) * rho)[:, None]
    candidates = clip_step(center, step)
    delta = candidates - center
    solved = mx.linalg.solve(chol_mx, delta.T, stream=mx.cpu).T
    dist2 = mx.sum(solved * solved, axis=1)
    scale = mx.where(
        dist2 > np.float32(radius2) * (1.0 + 1e-8),
        mx.sqrt(np.float32(radius2) / mx.maximum(dist2, np.float32(1e-12))),
        mx.ones_like(dist2),
    )
    out = clip_step(center, delta * scale[:, None])
    mx.eval(out)
    return from_mlx(out)


def fused_metric_candidates(
    z: np.ndarray,
    x_center: np.ndarray,
    cov_factor: np.ndarray,
    length: float,
) -> np.ndarray:
    mx = mlx_module()
    step = (to_mlx(z) @ to_mlx(cov_factor).T) * np.float32(length)
    out = clip_step(to_mlx(x_center).reshape(1, -1), step)
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
    mx = mlx_module()
    step = to_mlx(coeff) @ to_mlx(basis).T
    if sparse_scale != 0.0:
        step = step + np.float32(sparse_scale) * to_mlx(z)
    step = step * np.float32(length)
    out = clip_step(to_mlx(x_center).reshape(1, -1), step)
    mx.eval(out)
    return from_mlx(out)


def fused_ellipsoid_generate(
    z: np.ndarray,
    x_center: np.ndarray,
    chol,
    radius2: float,
) -> np.ndarray:
    mx = mlx_module()
    chol_mx = to_mlx(chol)
    center = to_mlx(x_center).reshape(1, -1)
    step = to_mlx(z) @ chol_mx.T
    candidates = clip_step(center, step)
    delta = candidates - center
    solved = mx.linalg.solve(chol_mx, delta.T, stream=mx.cpu).T
    dist2 = mx.sum(solved * solved, axis=1)
    scale = mx.where(
        dist2 > np.float32(radius2) * (1.0 + 1e-8),
        mx.sqrt(np.float32(radius2) / mx.maximum(dist2, np.float32(1e-12))),
        mx.ones_like(dist2),
    )
    out = clip_step(center, delta * scale[:, None])
    mx.eval(out)
    return from_mlx(out)
