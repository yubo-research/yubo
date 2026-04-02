"""Accelerated trust-region kernels (NumPy-in, NumPy-out).

Backend selection (auto, checked once on first call):
  1. MLX    — Apple Silicon GPU via Metal. Best on macOS.
  2. Triton — PyTorch CUDA + custom Triton kernels. Best on Linux GPU.
  3. JAX    — CUDA GPU on Linux, CPU fallback elsewhere.
  4. None   — callers fall back to NumPy (is_available() returns False).

Use set_backend("triton") or set_backend("jax") to force a specific backend.
All public functions accept NumPy arrays and return NumPy arrays.
"""

from __future__ import annotations

import numpy as np

from optimizer.trust_region_jax_mlx import (
    _mlx,
    _mlx_cholesky,
    _mlx_clip_step,
    _mlx_clip_to_unit_box,
    _mlx_low_rank_metric,
    _mlx_low_rank_step,
    _mlx_mahalanobis_sq,
    _mlx_mahalanobis_sq_from_cov,
    _mlx_matmul,
)

_BACKEND: str | None = None  # "mlx", "triton", "jax", or None
_FORCED: bool = False


def set_backend(name: str) -> None:
    """Force a specific backend. Call before any kernel use."""
    global _BACKEND, _FORCED
    allowed = {"mlx", "triton", "jax", ""}
    if name not in allowed:
        raise ValueError(f"Unknown backend {name!r}, expected one of {allowed}")
    _BACKEND = name
    _FORCED = True


def _detect_backend() -> str | None:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND if _BACKEND != "" else None
    try:
        import mlx.core as mx  # noqa: F401

        _BACKEND = "mlx"
        return _BACKEND
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            try:
                import triton  # noqa: F401

                _BACKEND = "triton"
                return _BACKEND
            except ImportError:
                pass
    except ImportError:
        pass
    try:
        import jax.numpy  # noqa: F401

        _BACKEND = "jax"
        return _BACKEND
    except ImportError:
        pass
    _BACKEND = ""
    return None


def is_available() -> bool:
    b = _detect_backend()
    return b is not None and b != ""


def backend_name() -> str:
    b = _detect_backend()
    return b if b else "none"


# ===================================================================
# JAX backend
# ===================================================================

_jnp = None
_lax = None
_jit = None


def _ensure_jax():
    global _jnp, _lax, _jit
    if _jnp is not None:
        return
    import jax.lax as lax
    import jax.numpy as jnp
    from jax import jit

    _jnp = jnp
    _lax = lax
    _jit = jit


def _jax_mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    _ensure_jax()
    d = _jnp.asarray(delta, dtype=_jnp.float32)
    ci = _jnp.asarray(cov_inv, dtype=_jnp.float32)
    out = _jax_mahalanobis_sq_jit(d, ci)
    return np.asarray(out, dtype=np.float64)


def _jax_mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    _ensure_jax()
    d = _jnp.asarray(delta, dtype=_jnp.float32)
    c = _jnp.asarray(cov, dtype=_jnp.float32)
    out = _jax_mahalanobis_sq_from_cov_jit(d, c)
    return np.asarray(out, dtype=np.float64)


def _jax_low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    _ensure_jax()
    c = _jnp.asarray(coeff, dtype=_jnp.float32)
    b = _jnp.asarray(basis, dtype=_jnp.float32)
    out = _jax_low_rank_step_jit(c, b)
    return np.asarray(out, dtype=np.float64)


def _jax_low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    _ensure_jax()
    d = _jnp.asarray(delta, dtype=_jnp.float32)
    b = _jnp.asarray(basis, dtype=_jnp.float32)
    be = _jnp.asarray(beta, dtype=_jnp.float32)
    out = _jax_low_rank_metric_jit(d, b, be, _jnp.float32(inv_alpha))
    return np.asarray(out, dtype=np.float64)


def _jax_clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    _ensure_jax()
    xc = _jnp.asarray(x_center, dtype=_jnp.float32).reshape(1, -1)
    s = _jnp.asarray(step, dtype=_jnp.float32)
    out = _jax_clip_to_unit_box_jit(xc, s)
    return np.asarray(out, dtype=np.float64)


def _jax_cholesky(cov: np.ndarray) -> np.ndarray:
    _ensure_jax()
    c = _jnp.asarray(cov, dtype=_jnp.float32)
    out = _jnp.linalg.cholesky(c)
    return np.asarray(out, dtype=np.float64)


def _jax_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _ensure_jax()
    ja = _jnp.asarray(a, dtype=_jnp.float32)
    jb = _jnp.asarray(b, dtype=_jnp.float32)
    out = ja @ jb
    return np.asarray(out, dtype=np.float64)


# JAX JIT stubs (self-replacing on first call)


def _jax_mahalanobis_sq_jit(delta, cov_inv):
    global _jax_mahalanobis_sq_jit

    @_jit
    def _fn(delta, cov_inv):
        return _jnp.einsum("nd,de,ne->n", delta, cov_inv, delta)

    _jax_mahalanobis_sq_jit = _fn
    return _fn(delta, cov_inv)


def _jax_mahalanobis_sq_from_cov_jit(delta, cov):
    global _jax_mahalanobis_sq_from_cov_jit
    import jax

    @_jit
    def _fn(delta, cov):
        tmp = jax.scipy.linalg.solve(cov, delta.T, assume_a="pos").T
        return _jnp.sum(delta * tmp, axis=1)

    _jax_mahalanobis_sq_from_cov_jit = _fn
    return _fn(delta, cov)


def _jax_low_rank_step_jit(coeff, basis):
    global _jax_low_rank_step_jit

    @_jit
    def _fn(coeff, basis):
        return _lax.dot(coeff, basis.T)

    _jax_low_rank_step_jit = _fn
    return _fn(coeff, basis)


def _jax_low_rank_metric_jit(delta, basis, beta, inv_alpha):
    global _jax_low_rank_metric_jit

    @_jit
    def _fn(delta, basis, beta, inv_alpha):
        proj = delta @ basis
        term1 = inv_alpha * _jnp.sum(delta * delta, axis=1)
        term2 = _jnp.sum(proj * proj * beta, axis=1)
        return term1 - term2

    _jax_low_rank_metric_jit = _fn
    return _fn(delta, basis, beta, inv_alpha)


def _jax_clip_to_unit_box_jit(x_center, step):
    global _jax_clip_to_unit_box_jit

    @_jit
    def _fn(x_center, step):
        pos = step > 0.0
        neg = step < 0.0
        safe_step = _jnp.where(step == 0.0, 1.0, step)
        t_pos = (1.0 - x_center) / safe_step
        t_neg = -x_center / safe_step
        t = _jnp.where(pos, t_pos, _jnp.where(neg, t_neg, _jnp.float32(1e30)))
        t_min = _jnp.min(t, axis=1)
        t_min = _jnp.clip(t_min, 0.0, 1.0)
        return x_center + step * t_min[:, None]

    _jax_clip_to_unit_box_jit = _fn
    return _fn(x_center, step)


# ===================================================================
# Triton / PyTorch CUDA backend
# ===================================================================

_torch = None
_triton_device = None


def _ensure_triton():
    global _torch, _triton_device
    if _torch is not None:
        return
    import torch

    _torch = torch
    _triton_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_torch(a: np.ndarray):
    _ensure_triton()
    return _torch.as_tensor(a.astype(np.float32), device=_triton_device)


def _from_torch(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


try:
    import triton as _triton_pkg
    import triton.language as tl

    @_triton_pkg.jit
    def _triton_mahalanobis_kernel(
        delta_ptr,
        cov_inv_ptr,
        out_ptr,
        n: tl.constexpr,
        d: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n:
            return
        acc = 0.0
        for j_start in range(0, d, BLOCK_D):
            j_offs = j_start + tl.arange(0, BLOCK_D)
            j_mask = j_offs < d
            delta_j = tl.load(delta_ptr + row * d + j_offs, mask=j_mask, other=0.0)
            for k_start in range(0, d, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < d
                delta_k = tl.load(delta_ptr + row * d + k_offs, mask=k_mask, other=0.0)
                cov_block = tl.load(
                    cov_inv_ptr + (k_offs[:, None] * d + j_offs[None, :]),
                    mask=k_mask[:, None] & j_mask[None, :],
                    other=0.0,
                )
                acc += tl.sum(delta_k[:, None] * cov_block * delta_j[None, :])
        tl.store(out_ptr + row, acc)

    @_triton_pkg.jit
    def _triton_matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

    @_triton_pkg.jit
    def _triton_low_rank_metric_kernel(
        delta_ptr,
        basis_ptr,
        beta_ptr,
        out_ptr,
        n: tl.constexpr,
        d: tl.constexpr,
        r: tl.constexpr,
        inv_alpha: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n:
            return
        term1 = 0.0
        for j_start in range(0, d, BLOCK_N):
            j_offs = j_start + tl.arange(0, BLOCK_N)
            j_mask = j_offs < d
            delta_j = tl.load(delta_ptr + row * d + j_offs, mask=j_mask, other=0.0)
            term1 += tl.sum(delta_j * delta_j)
        term1 = inv_alpha * term1
        term2 = 0.0
        for k_start in range(0, r, BLOCK_R):
            k_offs = k_start + tl.arange(0, BLOCK_R)
            k_mask = k_offs < r
            proj_block = tl.zeros((BLOCK_R,), dtype=tl.float32)
            for j_start in range(0, d, BLOCK_N):
                j_offs = j_start + tl.arange(0, BLOCK_N)
                j_mask = j_offs < d
                delta_j = tl.load(delta_ptr + row * d + j_offs, mask=j_mask, other=0.0)
                basis_jk = tl.load(
                    basis_ptr + (j_offs[:, None] * r + k_offs[None, :]),
                    mask=j_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                proj_block += tl.sum(delta_j[:, None] * basis_jk, axis=0)
            beta_k = tl.load(beta_ptr + k_offs, mask=k_mask, other=0.0)
            term2 += tl.sum(proj_block * proj_block * beta_k)
        tl.store(out_ptr + row, term1 - term2)

    _HAS_TRITON_KERNELS = True
except ImportError:
    _HAS_TRITON_KERNELS = False


def _triton_mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    _ensure_triton()
    d = _to_torch(delta)
    ci = _to_torch(cov_inv)
    if _HAS_TRITON_KERNELS:
        n, dim = d.shape
        out = _torch.empty(n, dtype=_torch.float32, device=d.device)
        BD = min(_triton_pkg.next_power_of_2(dim), 256)
        BK = min(64, dim)
        _triton_mahalanobis_kernel[(n,)](d, ci, out, n=n, d=dim, BLOCK_D=BD, BLOCK_K=BK)
    else:
        out = ((d @ ci) * d).sum(dim=1)
    return _from_torch(out)


def _triton_mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    _ensure_triton()
    d = _to_torch(delta)
    c = _to_torch(cov)
    ci = _torch.linalg.inv(c)
    if _HAS_TRITON_KERNELS:
        n, dim = d.shape
        out = _torch.empty(n, dtype=_torch.float32, device=d.device)
        BD = min(_triton_pkg.next_power_of_2(dim), 256)
        BK = min(64, dim)
        _triton_mahalanobis_kernel[(n,)](d, ci, out, n=n, d=dim, BLOCK_D=BD, BLOCK_K=BK)
    else:
        out = ((d @ ci) * d).sum(dim=1)
    return _from_torch(out)


def _triton_low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    _ensure_triton()
    c = _to_torch(coeff)
    b = _to_torch(basis)
    if _HAS_TRITON_KERNELS:
        n, r = c.shape
        dim = b.shape[0]
        out = _torch.empty(n, dim, dtype=_torch.float32, device=c.device)
        bt = b.T.contiguous()
        BM, BN, BK = 64, 64, 32
        grid = (_triton_pkg.cdiv(n, BM) * _triton_pkg.cdiv(dim, BN),)
        _triton_matmul_kernel[grid](c, bt, out, M=n, N=dim, K=r, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    else:
        out = c @ b.T
    return _from_torch(out)


def _triton_low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    _ensure_triton()
    d = _to_torch(delta)
    b = _to_torch(basis)
    be = _to_torch(beta)
    if _HAS_TRITON_KERNELS:
        n, dim = d.shape
        r = b.shape[1]
        d_c = d.contiguous()
        b_c = b.contiguous()
        out = _torch.empty(n, dtype=_torch.float32, device=d_c.device)
        BN = min(64, _triton_pkg.next_power_of_2(dim))
        BR = min(32, _triton_pkg.next_power_of_2(r))
        _triton_low_rank_metric_kernel[(n,)](d_c, b_c, be, out, n=n, d=dim, r=r, inv_alpha=inv_alpha, BLOCK_N=BN, BLOCK_R=BR)
    else:
        proj = d @ b
        out = inv_alpha * (d * d).sum(dim=1) - (proj * proj * be).sum(dim=1)
    return _from_torch(out)


def _torch_clip_step(xc, s):
    """Vectorized clip operating on torch tensors. Returns torch tensor."""
    safe = _torch.where(s == 0, _torch.tensor(1.0, device=s.device), s)
    tp = (1.0 - xc) / safe
    tn = -xc / safe
    t_all = _torch.where(s > 0, tp, _torch.where(s < 0, tn, _torch.tensor(1e30, device=s.device)))
    t_min = _torch.clamp(t_all.min(dim=1).values, 0.0, 1.0)
    return xc + s * t_min[:, None]


def _triton_clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    _ensure_triton()
    xc = _to_torch(x_center).reshape(1, -1)
    s = _to_torch(step)
    safe = _torch.where(s == 0, _torch.tensor(1.0, device=s.device), s)
    tp = (1.0 - xc) / safe
    tn = -xc / safe
    t_all = _torch.where(s > 0, tp, _torch.where(s < 0, tn, _torch.tensor(1e30, device=s.device)))
    t_min = _torch.clamp(t_all.min(dim=1).values, 0.0, 1.0)
    out = xc + s * t_min[:, None]
    return _from_torch(out)


def _triton_cholesky(cov: np.ndarray) -> np.ndarray:
    _ensure_triton()
    c = _to_torch(cov)
    out = _torch.linalg.cholesky(c)
    return _from_torch(out)


def _triton_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _ensure_triton()
    ta = _to_torch(a)
    tb = _to_torch(b)
    out = ta @ tb
    return _from_torch(out)


# ===================================================================
# Fused pipelines with on-device caching
# ===================================================================


class CovCache:
    """Caches Cholesky factor and inverse of a covariance matrix on-device.

    Uses a generation counter (bumped by the geometry model on each update)
    so that repeated calls with the same covariance skip recomputation.
    """

    __slots__ = ("_chol", "_inv", "_gen", "_backend")

    def __init__(self) -> None:
        self._chol = None
        self._inv = None
        self._gen: int = -1
        self._backend: str | None = None

    def update(self, cov: np.ndarray, gen: int = -1) -> None:
        if gen >= 0 and gen == self._gen and self._backend == _detect_backend():
            return
        b = _detect_backend()
        self._backend = b
        self._gen = gen
        if b == "mlx":
            mx = _mlx()
            c = mx.array(cov.astype(np.float32))
            self._chol = mx.linalg.cholesky(c, stream=mx.cpu)
            self._inv = mx.linalg.inv(c, stream=mx.cpu)
            mx.eval(self._chol, self._inv)
        else:
            self._chol = np.linalg.cholesky(cov)
            self._inv = np.linalg.inv(cov)

    def invalidate(self) -> None:
        self._chol = None
        self._inv = None
        self._gen = -1


def fused_ellipsoid_generate(
    z: np.ndarray,
    x_center: np.ndarray,
    covariance_matrix: np.ndarray,
    length: float,
    cache: CovCache,
    gen: int = -1,
) -> np.ndarray:
    """Fully fused ellipsoid: z @ chol.T → clip → mahalanobis → rescale bad → re-clip.

    Stays on-device (MLX/JAX) until the single final np conversion.
    """
    cache.update(covariance_matrix, gen=gen)
    radius2 = float(length) ** 2
    b = _detect_backend()
    if b == "mlx":
        mx = _mlx()
        z_mx = mx.array(z.astype(np.float32))
        xc = mx.array(x_center.astype(np.float32)).reshape(1, -1)
        step = z_mx @ cache._chol.T
        cands = _mlx_clip_step(xc, step)
        delta = cands - xc
        dist2 = mx.sum(delta * (delta @ cache._inv), axis=1)
        mx.eval(cands, dist2)
        dist2_np = np.asarray(dist2, dtype=np.float64)
        bad = np.where(dist2_np > radius2 * (1.0 + 1e-8))[0]
        if bad.size > 0:
            delta_np = np.asarray(delta, dtype=np.float64)
            scale = np.sqrt(radius2 / np.maximum(dist2_np[bad], 1e-12))
            delta_np[bad] *= scale.reshape(-1, 1)
            cands = _mlx_clip_step(xc, mx.array(delta_np.astype(np.float32)))
            mx.eval(cands)
        return np.asarray(cands, dtype=np.float64)
    if b in ("jax", "triton"):
        chol_np = cache._chol
        step_np = np.asarray(z) @ chol_np.T
        from optimizer.trust_region_math import _mahalanobis_sq, _ray_scale_to_unit_box

        cands = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + step_np)
        delta = cands - x_center.reshape(1, -1)
        dist2 = _mahalanobis_sq(delta, covariance_matrix)
        bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
        if bad.size > 0:
            scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
            delta[bad] *= scale.reshape(-1, 1)
            cands = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + delta)
        return cands
    chol = cache._chol
    step = z @ chol.T
    from optimizer.trust_region_math import _mahalanobis_sq, _ray_scale_to_unit_box

    cands = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + step)
    delta = cands - x_center.reshape(1, -1)
    dist2 = _mahalanobis_sq(delta, covariance_matrix)
    bad = np.where(dist2 > radius2 * (1.0 + 1e-8))[0]
    if bad.size > 0:
        scale = np.sqrt(radius2 / np.maximum(dist2[bad], 1e-12))
        delta[bad] *= scale.reshape(-1, 1)
        cands = _ray_scale_to_unit_box(x_center, x_center.reshape(1, -1) + delta)
    return cands


def whitened_sample(
    mask: np.ndarray,
    base: np.ndarray,
    u: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
) -> np.ndarray:
    """Fused mask → norm → normalize → radial scale on GPU.

    Inputs are NumPy arrays produced by the CPU RNG. The heavy element-wise
    math (5 array ops over (n, d)) is fused into a single GPU dispatch.
    """
    b = _detect_backend()
    if b == "mlx":
        mx = _mlx()
        m = mx.array(mask)
        z = mx.array(base.astype(np.float32)) * m
        norms = mx.linalg.norm(z, axis=1)
        safe_norms = mx.where(norms > 1e-12, norms, mx.array(1.0))
        v = z / safe_norms[:, None]
        u_mx = mx.array(u.astype(np.float32))
        if radial_mode == "boundary":
            rho = 0.8 + 0.2 * u_mx
        else:
            rho = mx.power(u_mx, 1.0 / max(num_dim, 1))
        out = np.float32(length) * rho[:, None] * v
        mx.eval(out)
        return np.asarray(out, dtype=np.float64)
    if b == "triton":
        _ensure_triton()
        m = _torch.tensor(mask, dtype=_torch.float32, device=_triton_device)
        z = _to_torch(base) * m
        norms = _torch.linalg.norm(z, dim=1)
        safe_norms = _torch.where(norms > 1e-12, norms, _torch.tensor(1.0, device=z.device))
        v = z / safe_norms[:, None]
        u_t = _to_torch(u)
        if radial_mode == "boundary":
            rho = 0.8 + 0.2 * u_t
        else:
            rho = _torch.pow(u_t, 1.0 / max(num_dim, 1))
        out = float(length) * rho[:, None] * v
        return _from_torch(out)
    return None  # caller falls back to numpy


def fused_metric_candidates(
    z: np.ndarray,
    x_center: np.ndarray,
    cov_factor: np.ndarray,
    length: float,
) -> np.ndarray:
    """Fused metric pipeline: z @ cov_factor.T * length → clip."""
    b = _detect_backend()
    if b == "mlx":
        mx = _mlx()
        z_mx = mx.array(z.astype(np.float32))
        f_mx = mx.array(cov_factor.T.astype(np.float32))
        xc = mx.array(x_center.astype(np.float32)).reshape(1, -1)
        step = (z_mx @ f_mx) * np.float32(length)
        out = _mlx_clip_step(xc, step)
        mx.eval(out)
        return np.asarray(out, dtype=np.float64)
    if b == "jax":
        _ensure_jax()
        z_j = _jnp.asarray(z, dtype=_jnp.float32)
        f_j = _jnp.asarray(cov_factor.T, dtype=_jnp.float32)
        xc_j = _jnp.asarray(x_center, dtype=_jnp.float32).reshape(1, -1)
        step = (z_j @ f_j) * _jnp.float32(length)
        return np.asarray(
            _jax_clip_to_unit_box_jit(xc_j, step),
            dtype=np.float64,
        )
    if b == "triton":
        _ensure_triton()
        z_t = _to_torch(z)
        f_t = _to_torch(cov_factor.T)
        xc_t = _to_torch(x_center).reshape(1, -1)
        step = (z_t @ f_t) * float(length)
        return _from_torch(_torch_clip_step(xc_t, step))
    from optimizer.trust_region_math import _apply_full_factor, _clip_to_unit_box

    step = _apply_full_factor(z, cov_factor) * float(length)
    return _clip_to_unit_box(np.asarray(x_center, dtype=float), step)


# ===================================================================
# Public API — dispatches to detected backend
# ===================================================================


def mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    b = _detect_backend()
    if b == "mlx":
        return _mlx_mahalanobis_sq(delta, cov_inv)
    if b == "triton":
        return _triton_mahalanobis_sq(delta, cov_inv)
    return _jax_mahalanobis_sq(delta, cov_inv)


def mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    b = _detect_backend()
    if b == "mlx":
        return _mlx_mahalanobis_sq_from_cov(delta, cov)
    if b == "triton":
        return _triton_mahalanobis_sq_from_cov(delta, cov)
    return _jax_mahalanobis_sq_from_cov(delta, cov)


def low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    b = _detect_backend()
    if b == "mlx":
        return _mlx_low_rank_step(coeff, basis)
    if b == "triton":
        return _triton_low_rank_step(coeff, basis)
    return _jax_low_rank_step(coeff, basis)


def low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    b = _detect_backend()
    if b == "mlx":
        return _mlx_low_rank_metric(delta, basis, beta, inv_alpha)
    if b == "triton":
        return _triton_low_rank_metric(delta, basis, beta, inv_alpha)
    return _jax_low_rank_metric(delta, basis, beta, inv_alpha)


def clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    b = _detect_backend()
    if b == "mlx":
        return _mlx_clip_to_unit_box(x_center, step)
    if b == "triton":
        return _triton_clip_to_unit_box(x_center, step)
    return _jax_clip_to_unit_box(x_center, step)


def ray_scale_to_unit_box(x_center: np.ndarray, x: np.ndarray) -> np.ndarray:
    center = np.asarray(x_center, dtype=float)
    return clip_to_unit_box(center, np.asarray(x, dtype=float) - center)


def cholesky(cov: np.ndarray) -> np.ndarray:
    b = _detect_backend()
    if b == "mlx":
        return _mlx_cholesky(cov)
    if b == "triton":
        return _triton_cholesky(cov)
    return _jax_cholesky(cov)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_name = _detect_backend()
    if b_name == "mlx":
        return _mlx_matmul(a, b)
    if b_name == "triton":
        return _triton_matmul(a, b)
    return _jax_matmul(a, b)
