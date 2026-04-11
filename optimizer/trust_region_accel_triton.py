"""Triton / PyTorch CUDA backend helpers for trust-region accelerated kernels."""

from __future__ import annotations

import numpy as np

torch_module = None
triton_device = None
ellipsoid_needs_inverse = False


def available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_loaded():
    global torch_module, triton_device
    if torch_module is not None:
        return
    import torch

    torch_module = torch
    triton_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch(a: np.ndarray):
    ensure_loaded()
    return torch_module.as_tensor(np.asarray(a, dtype=np.float32), device=triton_device)


def from_torch(a) -> np.ndarray:
    return a.detach().cpu().numpy().astype(np.float64)


def matmul_tensor(a, b):
    if has_triton_kernels and a.ndim == 2 and b.ndim == 2:
        m, k = a.shape
        b_rows, n = b.shape
        if k != b_rows:
            raise ValueError((a.shape, b.shape))
        out = torch_module.empty(m, n, dtype=torch_module.float32, device=a.device)
        block_m, block_n, block_k = 64, 64, 32
        grid = (triton_package.cdiv(m, block_m) * triton_package.cdiv(n, block_n),)
        matmul_kernel[grid](
            a.contiguous(),
            b.contiguous(),
            out,
            m=m,
            n=n,
            k=k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
        return out
    return a @ b


def mahalanobis_sq_tensor(delta, cov_inv):
    if has_triton_kernels:
        n, dim = delta.shape
        out = torch_module.empty(n, dtype=torch_module.float32, device=delta.device)
        block_d = min(triton_package.next_power_of_2(dim), 256)
        block_k = min(64, dim)
        mahalanobis_kernel[(n,)](
            delta.contiguous(),
            cov_inv.contiguous(),
            out,
            n=n,
            d=dim,
            block_d=block_d,
            block_k=block_k,
        )
        return out
    return (matmul_tensor(delta, cov_inv) * delta).sum(dim=1)


def mahalanobis_sq_from_factor_tensor(delta, chol):
    solved = torch_module.linalg.solve_triangular(chol, delta.T, upper=False).T
    return (solved * solved).sum(dim=1)


try:
    import triton as triton_package
    import triton.language as tl

    @triton_package.jit
    def mahalanobis_kernel(
        delta_ptr,
        cov_inv_ptr,
        out_ptr,
        n: tl.constexpr,
        d: tl.constexpr,
        block_d: tl.constexpr,
        block_k: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n:
            return
        acc = 0.0
        for j_start in range(0, d, block_d):
            j_offs = j_start + tl.arange(0, block_d)
            j_mask = j_offs < d
            delta_j = tl.load(delta_ptr + row * d + j_offs, mask=j_mask, other=0.0)
            for k_start in range(0, d, block_k):
                k_offs = k_start + tl.arange(0, block_k)
                k_mask = k_offs < d
                delta_k = tl.load(delta_ptr + row * d + k_offs, mask=k_mask, other=0.0)
                cov_block = tl.load(
                    cov_inv_ptr + (k_offs[:, None] * d + j_offs[None, :]),
                    mask=k_mask[:, None] & j_mask[None, :],
                    other=0.0,
                )
                acc += tl.sum(delta_k[:, None] * cov_block * delta_j[None, :])
        tl.store(out_ptr + row, acc)

    @triton_package.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(n, block_n)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        a_ptrs = a_ptr + (offs_m[:, None] * k + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * n + offs_n[None, :])
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for _ in range(0, k, block_k):
            a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
            b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += block_k
            b_ptrs += block_k * n
        c_ptrs = c_ptr + (offs_m[:, None] * n + offs_n[None, :])
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        tl.store(c_ptrs, acc, mask=c_mask)

    @triton_package.jit
    def low_rank_metric_kernel(
        delta_ptr,
        basis_ptr,
        beta_ptr,
        out_ptr,
        n: tl.constexpr,
        d: tl.constexpr,
        r: tl.constexpr,
        inv_alpha: tl.constexpr,
        block_n: tl.constexpr,
        block_r: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n:
            return
        term1 = 0.0
        for j_start in range(0, d, block_n):
            j_offs = j_start + tl.arange(0, block_n)
            j_mask = j_offs < d
            delta_j = tl.load(delta_ptr + row * d + j_offs, mask=j_mask, other=0.0)
            term1 += tl.sum(delta_j * delta_j)
        term1 = inv_alpha * term1
        term2 = 0.0
        for k_start in range(0, r, block_r):
            k_offs = k_start + tl.arange(0, block_r)
            k_mask = k_offs < r
            proj_block = tl.zeros((block_r,), dtype=tl.float32)
            for j_start in range(0, d, block_n):
                j_offs = j_start + tl.arange(0, block_n)
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

    has_triton_kernels = True
except ImportError:
    has_triton_kernels = False


def clip_step(x_center, step):
    safe_step = torch_module.where(step == 0, torch_module.tensor(1.0, device=step.device), step)
    t_pos = (1.0 - x_center) / safe_step
    t_neg = -x_center / safe_step
    t = torch_module.where(
        step > 0,
        t_pos,
        torch_module.where(step < 0, t_neg, torch_module.tensor(1e30, device=step.device)),
    )
    t_min = torch_module.clamp(t.min(dim=1).values, 0.0, 1.0)
    return x_center + step * t_min[:, None]


def factorize_cov(cov: np.ndarray, *, need_inv: bool = True):
    cov_torch = to_torch(cov)
    inv = torch_module.linalg.inv(cov_torch) if need_inv else None
    return torch_module.linalg.cholesky(cov_torch), inv


def mahalanobis_sq(delta: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    return from_torch(mahalanobis_sq_tensor(to_torch(delta), to_torch(cov_inv)))


def mahalanobis_sq_from_factor(delta: np.ndarray, chol: np.ndarray) -> np.ndarray:
    return from_torch(mahalanobis_sq_from_factor_tensor(to_torch(delta), to_torch(chol)))


def mahalanobis_sq_from_cov(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov_torch = to_torch(cov)
    delta_torch = to_torch(delta)
    chol = torch_module.linalg.cholesky(cov_torch)
    return from_torch(mahalanobis_sq_from_factor_tensor(delta_torch, chol))


def low_rank_step(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coeff_torch = to_torch(coeff)
    basis_torch = to_torch(basis)
    return from_torch(matmul_tensor(coeff_torch, basis_torch.T))


def low_rank_step_with_sparse(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    scale: float,
) -> np.ndarray:
    out = matmul_tensor(to_torch(coeff), to_torch(basis).T)
    if scale != 0.0:
        out = out + float(scale) * to_torch(z)
    return from_torch(out)


def low_rank_metric(
    delta: np.ndarray,
    basis: np.ndarray,
    beta: np.ndarray,
    inv_alpha: float,
) -> np.ndarray:
    delta_torch = to_torch(delta)
    basis_torch = to_torch(basis)
    beta_torch = to_torch(beta)
    if has_triton_kernels:
        n, dim = delta_torch.shape
        rank = basis_torch.shape[1]
        out = torch_module.empty(n, dtype=torch_module.float32, device=delta_torch.device)
        block_n = min(64, triton_package.next_power_of_2(dim))
        block_r = min(32, triton_package.next_power_of_2(rank))
        low_rank_metric_kernel[(n,)](
            delta_torch.contiguous(),
            basis_torch.contiguous(),
            beta_torch,
            out,
            n=n,
            d=dim,
            r=rank,
            inv_alpha=inv_alpha,
            block_n=block_n,
            block_r=block_r,
        )
    else:
        proj = delta_torch @ basis_torch
        out = inv_alpha * (delta_torch * delta_torch).sum(dim=1) - (proj * proj * beta_torch).sum(dim=1)
    return from_torch(out)


def clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    return from_torch(clip_step(to_torch(x_center).reshape(1, -1), to_torch(step)))


def cholesky(cov: np.ndarray) -> np.ndarray:
    chol, _ = factorize_cov(cov)
    return from_torch(chol)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return from_torch(matmul_tensor(to_torch(a), to_torch(b)))


def whitened_sample(
    z_tilde: np.ndarray,
    u: np.ndarray,
    length: float,
    radial_mode: str,
    num_dim: int,
) -> np.ndarray:
    z = to_torch(z_tilde)
    norms = torch_module.linalg.norm(z, dim=1)
    safe_norms = torch_module.where(norms > 1e-12, norms, torch_module.tensor(1.0, device=z.device))
    directions = z / safe_norms[:, None]
    u_torch = to_torch(u)
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u_torch
    else:
        rho = torch_module.pow(u_torch, 1.0 / max(num_dim, 1))
    return from_torch(float(length) * rho[:, None] * directions)


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
    z = to_torch(z_tilde)
    norms = torch_module.linalg.norm(z, dim=1)
    safe_norms = torch_module.where(norms > 1e-12, norms, torch_module.tensor(1.0, device=z.device))
    directions = z / safe_norms[:, None]
    u_torch = to_torch(u)
    if radial_mode == "boundary":
        rho = 0.8 + 0.2 * u_torch
    else:
        rho = torch_module.pow(u_torch, 1.0 / max(num_dim, 1))
    whitened = float(length) * rho[:, None] * directions
    center = to_torch(x_center).reshape(1, -1)
    step = matmul_tensor(whitened, chol.T)
    candidates = clip_step(center, step)
    delta = candidates - center
    dist2 = mahalanobis_sq_from_factor_tensor(delta, chol)
    scale = torch_module.where(
        dist2 > float(radius2) * (1.0 + 1e-8),
        torch_module.sqrt(float(radius2) / torch_module.clamp(dist2, min=1e-12)),
        torch_module.ones_like(dist2),
    )
    return from_torch(clip_step(center, delta * scale[:, None]))


def fused_metric_candidates(
    z: np.ndarray,
    x_center: np.ndarray,
    cov_factor: np.ndarray,
    length: float,
) -> np.ndarray:
    step = matmul_tensor(to_torch(z), to_torch(cov_factor).T) * float(length)
    return from_torch(clip_step(to_torch(x_center).reshape(1, -1), step))


def fused_low_rank_candidates(
    coeff: np.ndarray,
    basis: np.ndarray,
    z: np.ndarray,
    sparse_scale: float,
    x_center: np.ndarray,
    length: float,
) -> np.ndarray:
    step = matmul_tensor(to_torch(coeff), to_torch(basis).T)
    if sparse_scale != 0.0:
        step = step + float(sparse_scale) * to_torch(z)
    step = step * float(length)
    return from_torch(clip_step(to_torch(x_center).reshape(1, -1), step))


def fused_ellipsoid_generate(
    z: np.ndarray,
    x_center: np.ndarray,
    chol,
    radius2: float,
) -> np.ndarray:
    center = to_torch(x_center).reshape(1, -1)
    step = matmul_tensor(to_torch(z), chol.T)
    candidates = clip_step(center, step)
    delta = candidates - center
    dist2 = mahalanobis_sq_from_factor_tensor(delta, chol)
    scale = torch_module.where(
        dist2 > float(radius2) * (1.0 + 1e-8),
        torch_module.sqrt(float(radius2) / torch_module.clamp(dist2, min=1e-12)),
        torch_module.ones_like(dist2),
    )
    return from_torch(clip_step(center, delta * scale[:, None]))
