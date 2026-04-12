"""JAX backend helpers for trust-region accelerated kernels."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

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

jnp = None
jit = None
jsp_linalg = None
ellipsoid_needs_inverse = False


def available() -> bool:
    try:
        import jax.numpy  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_loaded():
    global jnp, jit, jsp_linalg
    if jnp is not None:
        return
    import jax.numpy as loaded_jnp
    import jax.scipy.linalg as loaded_jsp_linalg
    from jax import jit as loaded_jit

    jnp = loaded_jnp
    jit = loaded_jit
    jsp_linalg = loaded_jsp_linalg


def _ops() -> BackendOps:
    ensure_loaded()
    return BackendOps(
        sum_rows=lambda x: jnp.sum(x, axis=1),
        min_rows=lambda x: jnp.min(x, axis=1),
        norm_rows=lambda x: jnp.linalg.norm(x, axis=1),
        where=jnp.where,
        power=jnp.power,
        sqrt=jnp.sqrt,
        matmul=lambda a, b: a @ b,
        solve=lambda a, b: jsp_linalg.solve(a, b, assume_a="pos"),
        solve_triangular=lambda a, b: jsp_linalg.solve_triangular(a, b, lower=True),
    )


def mahalanobis_formula(delta, cov_inv):
    return _mahalanobis_sq_formula(delta, cov_inv, _ops())


def clip_step(x_center, step):
    return _clip_step_formula(x_center, step, _ops())


def mahalanobis_from_cov_formula(delta, cov):
    return _mahalanobis_from_cov_formula(delta, cov, _ops())


def mahalanobis_from_factor_formula(delta, chol):
    return _mahalanobis_from_factor_formula(delta, chol, _ops())


def low_rank_step_formula(coeff, basis):
    return _low_rank_step_formula(coeff, basis, _ops())


def low_rank_step_with_sparse_formula(coeff, basis, z, scale):
    return _low_rank_step_with_sparse_formula(coeff, basis, z, scale, _ops())


def low_rank_metric_formula(delta, basis, beta, inv_alpha):
    return _low_rank_metric_formula(delta, basis, beta, inv_alpha, _ops())


def matmul_formula(a, b):
    return _ops().matmul(a, b)


def whitened_sample_formula(z_tilde, u, length, boundary, inv_dim):
    return _whitened_sample_formula(z_tilde, u, length, boundary, inv_dim, _ops())


def fused_metric_candidates_formula(z, x_center, cov_factor_t, length):
    return _fused_metric_candidates_formula(z, x_center, cov_factor_t, length, _ops())


def fused_low_rank_candidates_formula(coeff, basis, z, sparse_scale, x_center, length):
    return _fused_low_rank_candidates_formula(coeff, basis, z, sparse_scale, x_center, length, _ops())


def fused_ellipsoid_generate_formula(z, x_center, chol, radius2):
    return _fused_ellipsoid_generate_formula(z, x_center, chol, radius2, _ops())


def fused_whitened_ellipsoid_candidates_formula(
    z_tilde,
    u,
    x_center,
    chol,
    length,
    boundary,
    inv_dim,
    radius2,
):
    return _fused_whitened_ellipsoid_candidates_formula(
        z_tilde,
        u,
        x_center,
        chol,
        length,
        boundary,
        inv_dim,
        radius2,
        _ops(),
    )


@lru_cache(maxsize=1)
def kernels() -> SimpleNamespace:
    ensure_loaded()
    return SimpleNamespace(
        mahalanobis_sq=jit(mahalanobis_formula),
        mahalanobis_sq_from_cov=jit(mahalanobis_from_cov_formula),
        mahalanobis_sq_from_factor=jit(mahalanobis_from_factor_formula),
        low_rank_step=jit(low_rank_step_formula),
        low_rank_step_with_sparse=jit(low_rank_step_with_sparse_formula),
        low_rank_metric=jit(low_rank_metric_formula),
        clip_to_unit_box=jit(clip_step),
        cholesky=jit(jnp.linalg.cholesky),
        matmul=jit(matmul_formula),
        whitened_sample=jit(whitened_sample_formula),
        fused_whitened_ellipsoid_candidates=jit(fused_whitened_ellipsoid_candidates_formula),
        fused_metric_candidates=jit(fused_metric_candidates_formula),
        fused_low_rank_candidates=jit(fused_low_rank_candidates_formula),
        fused_ellipsoid_generate=jit(fused_ellipsoid_generate_formula),
    )


def factorize_cov(cov: np.ndarray, *, need_inv: bool = False):
    ensure_loaded()
    cov_jax = jnp.asarray(cov, dtype=jnp.float32)
    inv = jnp.linalg.inv(cov_jax) if need_inv else None
    return kernels().cholesky(cov_jax), inv


def mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    ensure_loaded()
    out = kernels().mahalanobis_sq(
        jnp.asarray(delta, dtype=jnp.float32),
        jnp.asarray(cov_inv, dtype=jnp.float32),
    )
    return np.asarray(out, dtype=np.float64)


def mahalanobis_sq_from_factor(delta: np.ndarray, chol: np.ndarray) -> np.ndarray:
    ensure_loaded()
    out = kernels().mahalanobis_sq_from_factor(
        jnp.asarray(delta, dtype=jnp.float32),
        jnp.asarray(chol, dtype=jnp.float32),
    )
    return np.asarray(out, dtype=np.float64)


def mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    ensure_loaded()
    chol, _ = factorize_cov(cov, need_inv=False)
    return mahalanobis_sq_from_factor(delta, chol)


def low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    ensure_loaded()
    out = kernels().low_rank_step(
        jnp.asarray(coeff, dtype=jnp.float32),
        jnp.asarray(basis, dtype=jnp.float32),
    )
    return np.asarray(out, dtype=np.float64)


def low_rank_step_with_sparse(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> np.ndarray:
    ensure_loaded()
    out = kernels().low_rank_step_with_sparse(
        jnp.asarray(coeff, dtype=jnp.float32),
        jnp.asarray(basis, dtype=jnp.float32),
        jnp.asarray(z, dtype=jnp.float32),
        jnp.float32(scale),
    )
    return np.asarray(out, dtype=np.float64)


def low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    ensure_loaded()
    out = kernels().low_rank_metric(
        jnp.asarray(delta, dtype=jnp.float32),
        jnp.asarray(basis, dtype=jnp.float32),
        jnp.asarray(beta, dtype=jnp.float32),
        jnp.float32(inv_alpha),
    )
    return np.asarray(out, dtype=np.float64)


def clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    ensure_loaded()
    out = kernels().clip_to_unit_box(
        jnp.asarray(x_center, dtype=jnp.float32).reshape(1, -1),
        jnp.asarray(step, dtype=jnp.float32),
    )
    return np.asarray(out, dtype=np.float64)


def cholesky(cov: np.ndarray) -> np.ndarray:
    ensure_loaded()
    out = kernels().cholesky(jnp.asarray(cov, dtype=jnp.float32))
    return np.asarray(out, dtype=np.float64)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ensure_loaded()
    out = kernels().matmul(
        jnp.asarray(a, dtype=jnp.float32),
        jnp.asarray(b, dtype=jnp.float32),
    )
    return np.asarray(out, dtype=np.float64)


def whitened_sample(
    z_tilde: np.ndarray,
    u: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
) -> np.ndarray:
    ensure_loaded()
    out = kernels().whitened_sample(
        jnp.asarray(z_tilde, dtype=jnp.float32),
        jnp.asarray(u, dtype=jnp.float32),
        jnp.float32(length),
        jnp.asarray(radial_mode == "boundary"),
        jnp.float32(1.0 / max(num_dim, 1)),
    )
    return np.asarray(out, dtype=np.float64)


def fused_metric_candidates(
    z: np.ndarray,
    x_center: np.ndarray,
    cov_factor: np.ndarray,
    length: float,
) -> np.ndarray:
    ensure_loaded()
    out = kernels().fused_metric_candidates(
        jnp.asarray(z, dtype=jnp.float32),
        jnp.asarray(x_center, dtype=jnp.float32).reshape(1, -1),
        jnp.asarray(cov_factor.T, dtype=jnp.float32),
        jnp.float32(length),
    )
    return np.asarray(out, dtype=np.float64)


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
    ensure_loaded()
    out = kernels().fused_whitened_ellipsoid_candidates(
        jnp.asarray(z_tilde, dtype=jnp.float32),
        jnp.asarray(u, dtype=jnp.float32),
        jnp.asarray(x_center, dtype=jnp.float32).reshape(1, -1),
        chol,
        jnp.float32(length),
        jnp.asarray(radial_mode == "boundary"),
        jnp.float32(1.0 / max(num_dim, 1)),
        jnp.float32(radius2),
    )
    return np.asarray(out, dtype=np.float64)


def fused_low_rank_candidates(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    sparse_scale: float,
    x_center: np.ndarray,
    length: float,
) -> np.ndarray:
    ensure_loaded()
    out = kernels().fused_low_rank_candidates(
        jnp.asarray(coeff, dtype=jnp.float32),
        jnp.asarray(basis, dtype=jnp.float32),
        jnp.asarray(z, dtype=jnp.float32),
        jnp.float32(sparse_scale),
        jnp.asarray(x_center, dtype=jnp.float32).reshape(1, -1),
        jnp.float32(length),
    )
    return np.asarray(out, dtype=np.float64)


def fused_ellipsoid_generate(
    z: np.ndarray,
    x_center: np.ndarray,
    chol,
    radius2: float,
) -> np.ndarray:
    ensure_loaded()
    out = kernels().fused_ellipsoid_generate(
        jnp.asarray(z, dtype=jnp.float32),
        jnp.asarray(x_center, dtype=jnp.float32).reshape(1, -1),
        chol,
        jnp.float32(radius2),
    )
    return np.asarray(out, dtype=np.float64)
