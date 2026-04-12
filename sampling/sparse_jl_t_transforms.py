import math

import torch
from torch import nn

from .sparse_jl_t_accum import accumulate_into, accumulate_into_wr, accumulate_noise_into
from .sparse_jl_t_hash import compute_rows_and_signs, compute_rows_and_signs_wr


def block_sparse_hash_scatter_from_nz_t(
    nz_indices: torch.Tensor,
    nz_values: torch.Tensor,
    d: int,
    s: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Scatter JL contributions from sparse (index, value) pairs."""
    y = torch.zeros(d, dtype=dtype, device=device)
    if nz_indices.numel() == 0:
        return y
    inv_sqrt_s = 1.0 / math.sqrt(s)
    rows, signs = compute_rows_and_signs(nz_indices.to(torch.int64), d, s, seed, device)
    contrib = signs * nz_values.to(dtype).unsqueeze(1) * inv_sqrt_s
    y.scatter_add_(0, rows.reshape(-1), contrib.reshape(-1).to(dtype))
    return y


def block_sparse_jl_noise_from_seed(
    *,
    num_dim_ambient: int,
    d: int,
    s: int = 4,
    jl_seed: int = 42,
    noise_seed: int,
    sigma: float,
    prob: float | None = None,
    chunk_size: int = 2**16,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sparse JL transform of a seeded noise vector without O(D) allocation."""
    if s > d:
        raise ValueError("s must be <= d")
    if device is None:
        device = torch.device("cpu")
    y = torch.zeros(d, dtype=dtype, device=device)
    accumulate_noise_into(
        y,
        int(num_dim_ambient),
        d=d,
        s=s,
        seed_jl=int(jl_seed),
        seed_noise=int(noise_seed),
        sigma=float(sigma),
        prob=None if prob is None else float(prob),
        chunk_size=int(chunk_size),
    )
    return y


def block_sparse_jl_noise_from_seed_wr(
    *,
    num_dim_ambient: int,
    d: int,
    s: int = 4,
    jl_seed: int = 42,
    noise_seed: int,
    sigma: float,
    prob: float | None = None,
    chunk_size: int = 2**16,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """With-replacement variant of `block_sparse_jl_noise_from_seed`."""
    if s > d:
        raise ValueError("s must be <= d")
    if device is None:
        device = torch.device("cpu")
    y = torch.zeros(d, dtype=dtype, device=device)
    if num_dim_ambient <= 0:
        return y
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    inv_sqrt_s = 1.0 / math.sqrt(s)
    g = torch.Generator(device=str(device))
    g.manual_seed(int(noise_seed))

    for start in range(0, int(num_dim_ambient), int(chunk_size)):
        end = min(start + int(chunk_size), int(num_dim_ambient))
        n = int(end - start)
        if prob is None:
            noise = torch.randn((n,), device=device, dtype=dtype, generator=g).mul_(float(sigma))
        else:
            u = torch.rand((n,), device=device, generator=g)
            mask = u < float(prob)
            noise = torch.randn((n,), device=device, dtype=dtype, generator=g).mul_(float(sigma))
            noise.mul_(mask)

        global_idx = torch.arange(start, end, dtype=torch.int64, device=device)
        rows, signs = compute_rows_and_signs_wr(global_idx, d, s, jl_seed, device)
        contrib = signs * noise.unsqueeze(1) * inv_sqrt_s
        y.scatter_add_(0, rows.reshape(-1), contrib.reshape(-1).to(dtype))
    return y


def block_sparse_jl_transform_t(x: torch.Tensor, d: int, s: int = 4, seed: int = 42) -> torch.Tensor:
    """Sparse JL transform of a 1-D tensor."""
    assert x.ndim == 1
    assert d > 0
    assert s > 0
    if s > d:
        raise ValueError("s must be <= d")
    y = torch.zeros(d, dtype=x.dtype, device=x.device)
    accumulate_into(y, x, offset=0, d=d, s=s, seed=seed)
    return y


def block_sparse_jl_transform_module(module: nn.Module, d: int, s: int = 4, seed: int = 42) -> torch.Tensor:
    """Sparse JL transform of an nn.Module's flattened parameter vector."""
    if s > d:
        raise ValueError("s must be <= d")
    params_iter = module.parameters()
    first_param = next(params_iter, None)
    if first_param is None:
        return torch.zeros(d)
    dtype = first_param.dtype
    device = first_param.device
    y = torch.zeros(d, dtype=dtype, device=device)

    flat = first_param.detach().reshape(-1)
    accumulate_into(y, flat, 0, d, s, seed)
    offset = flat.numel()

    for param in params_iter:
        flat = param.detach().reshape(-1)
        accumulate_into(y, flat, offset, d, s, seed)
        offset += flat.numel()
    return y


def block_sparse_jl_transform_module_wr(module: nn.Module, d: int, s: int = 4, seed: int = 42) -> torch.Tensor:
    """With-replacement variant of `block_sparse_jl_transform_module`."""
    if s > d:
        raise ValueError("s must be <= d")
    params_iter = module.parameters()
    first_param = next(params_iter, None)
    if first_param is None:
        return torch.zeros(d)
    dtype = first_param.dtype
    device = first_param.device
    y = torch.zeros(d, dtype=dtype, device=device)

    flat = first_param.detach().reshape(-1)
    accumulate_into_wr(y, flat, 0, d, s, seed)
    offset = flat.numel()

    for param in params_iter:
        flat = param.detach().reshape(-1)
        accumulate_into_wr(y, flat, offset, d, s, seed)
        offset += flat.numel()
    return y


def block_sparse_jl_transform_module_to_cpu_wr(
    module: nn.Module,
    d: int,
    s: int = 4,
    seed: int = 42,
    chunk_size: int = 2**16,
) -> torch.Tensor:
    """With-replacement module embedding, accumulated on CPU in chunks."""
    if s > d:
        raise ValueError("s must be <= d")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    y = torch.zeros(d, dtype=torch.float32, device=torch.device("cpu"))
    inv_sqrt_s = 1.0 / math.sqrt(s)
    offset = 0
    for param in module.parameters():
        flat = param.detach().reshape(-1)
        n = int(flat.numel())
        for start in range(0, n, int(chunk_size)):
            end = min(start + int(chunk_size), n)
            sub = flat[start:end].to(device=torch.device("cpu"), dtype=torch.float32)
            global_idx = torch.arange(
                offset + start,
                offset + end,
                dtype=torch.int64,
                device=torch.device("cpu"),
            )
            rows, signs = compute_rows_and_signs_wr(global_idx, d, s, seed, torch.device("cpu"))
            contrib = signs * sub.unsqueeze(1) * inv_sqrt_s
            y.scatter_add_(0, rows.reshape(-1), contrib.reshape(-1).to(y.dtype))
        offset += n
    return y
