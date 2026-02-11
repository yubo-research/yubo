import math

import torch
from torch import nn

# Splitmix64 constants (signed int64 for PyTorch compatibility)
_GOLDEN = 0x9E3779B97F4A7C15 - (1 << 64)
_MIX_M1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX_M2 = 0x94D049BB133111EB - (1 << 64)

_CHUNK_SIZE = 1 << 20  # 1M elements per sub-chunk
_MASK64 = (1 << 64) - 1


def _wrap_int64(val: int) -> int:
    """Wrap an arbitrary-precision Python int to signed int64 range."""
    val = val & _MASK64
    return val - (1 << 64) if val >= (1 << 63) else val


def _vmix64_inplace(z: torch.Tensor, tmp: torch.Tensor) -> None:
    """In-place splitmix64 finalizer. Modifies z, uses tmp as scratch."""
    torch.bitwise_right_shift(z, 30, out=tmp)
    tmp.bitwise_and_((1 << 34) - 1)
    z.bitwise_xor_(tmp)
    z.mul_(_MIX_M1)
    torch.bitwise_right_shift(z, 27, out=tmp)
    tmp.bitwise_and_((1 << 37) - 1)
    z.bitwise_xor_(tmp)
    z.mul_(_MIX_M2)
    torch.bitwise_right_shift(z, 31, out=tmp)
    tmp.bitwise_and_((1 << 33) - 1)
    z.bitwise_xor_(tmp)


def _vmix64(z: torch.Tensor) -> torch.Tensor:
    """Vectorized splitmix64 output finalizer (allocating version)."""
    z = z ^ ((z >> 30) & ((1 << 34) - 1))
    z = z * _MIX_M1
    z = z ^ ((z >> 27) & ((1 << 37) - 1))
    z = z * _MIX_M2
    z = z ^ ((z >> 31) & ((1 << 33) - 1))
    return z


def _compute_rows_and_signs(
    global_indices: torch.Tensor,
    d: int,
    s: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target rows (without replacement) and signs for coordinates.

    Uses Fisher-Yates-style selection: slot k draws from d-k candidates
    and remaps past previously chosen rows.

    Returns rows (N, s) int64 and signs (N, s) float32.
    """
    n = global_indices.shape[0]
    rows = torch.empty(n, s, dtype=torch.int64, device=device)
    signs = torch.empty(n, s, dtype=torch.float32, device=device)

    # Pre-allocate work buffers for in-place hash
    h = torch.empty(n, dtype=torch.int64, device=device)
    tmp = torch.empty(n, dtype=torch.int64, device=device)
    cand = torch.empty(n, dtype=torch.int64, device=device)

    for k in range(s):
        offset = _wrap_int64(seed * _GOLDEN + k * _GOLDEN)
        torch.add(global_indices, offset, out=h)
        _vmix64_inplace(h, tmp)

        torch.remainder(h, d - k, out=cand)

        # Sign from bit 32 (independent of lower bits used by modulo for small d)
        torch.bitwise_right_shift(h, 32, out=tmp)
        tmp.bitwise_and_(1)
        signs[:, k] = torch.where(tmp == 1, 1.0, -1.0)

        # Fisher-Yates: remap candidate past previously chosen rows (sorted)
        if k > 0:
            prev_sorted, _ = torch.sort(rows[:, :k], dim=1)
            for p in range(k):
                cand.add_((cand >= prev_sorted[:, p]).long())

        rows[:, k] = cand

    return rows, signs


def _accumulate_into(
    y: torch.Tensor,
    flat_values: torch.Tensor,
    offset: int,
    d: int,
    s: int,
    seed: int,
) -> None:
    """Accumulate JL contributions from flat_values into y (in-place)."""
    n = flat_values.numel()
    if n == 0:
        return
    device = y.device
    dtype = y.dtype
    inv_sqrt_s = 1.0 / math.sqrt(s)

    for start in range(0, n, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n)
        sub = flat_values[start:end]
        global_idx = torch.arange(offset + start, offset + end, dtype=torch.int64, device=device)
        rows, signs = _compute_rows_and_signs(global_idx, d, s, seed, device)
        contrib = signs * sub.to(dtype).unsqueeze(1) * inv_sqrt_s
        y.scatter_add_(0, rows.reshape(-1), contrib.reshape(-1).to(dtype))


def _block_sparse_hash_scatter_from_nz_t(
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
    rows, signs = _compute_rows_and_signs(nz_indices.to(torch.int64), d, s, seed, device)
    contrib = signs * nz_values.to(dtype).unsqueeze(1) * inv_sqrt_s
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
    _accumulate_into(y, x, offset=0, d=d, s=s, seed=seed)
    return y


def block_sparse_jl_transform_module(module: nn.Module, d: int, s: int = 4, seed: int = 42) -> torch.Tensor:
    """Sparse JL transform of an nn.Module's flattened parameter vector.

    Iterates over parameters without copying the full D-vector.
    Memory: O(d + chunk_size * s), independent of total D.
    """
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
    _accumulate_into(y, flat, 0, d, s, seed)
    offset = flat.numel()

    for param in params_iter:
        flat = param.detach().reshape(-1)
        _accumulate_into(y, flat, offset, d, s, seed)
        offset += flat.numel()
    return y
