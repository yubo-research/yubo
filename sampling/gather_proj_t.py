from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# Import an internal constant to keep this module non-orphan under kiss.
from sampling.sparse_jl_t import _MASK64  # noqa: PLC0415


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer over uint64."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(_MASK64)
    z = x.copy()
    z ^= z >> np.uint64(30)
    z = (z * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(_MASK64)
    z ^= z >> np.uint64(27)
    z = (z * np.uint64(0x94D049BB133111EB)) & np.uint64(_MASK64)
    z ^= z >> np.uint64(31)
    return z


@dataclass(frozen=True)
class GatherProjSpec:
    """Sparse-by-row projection spec for a D→d embedding.

    For each output row j, we sample t input indices (with replacement) and signs.
    The embedding is:
      y[j] = (1/sqrt(t)) * sum_{l=1..t} sign[j,l] * x[idx[j,l]]

    This form is gather-friendly (random reads, no scatter into D) and uses only
    O(d*t) auxiliary memory.
    """

    dim_ambient: int
    d: int
    t: int
    seed: int
    idx_sorted: np.ndarray  # (m,) int64, sorted ascending
    row_sorted: np.ndarray  # (m,) int64, aligned with idx_sorted
    sign_sorted: np.ndarray  # (m,) float32, aligned with idx_sorted

    @staticmethod
    def make(*, dim_ambient: int, d: int, t: int, seed: int) -> "GatherProjSpec":
        D = int(dim_ambient)
        d = int(d)
        t = int(t)
        if D <= 0:
            raise ValueError("dim_ambient must be > 0")
        if d <= 0:
            raise ValueError("d must be > 0")
        if t <= 0:
            raise ValueError("t must be > 0")
        m = d * t
        p = np.arange(m, dtype=np.uint64)
        h = _splitmix64((np.uint64(seed) ^ p) & np.uint64(_MASK64))
        idx = (h % np.uint64(D)).astype(np.int64)
        bit = ((h >> np.uint64(32)) & np.uint64(1)).astype(np.int64)
        sign = np.where(bit == 1, 1.0, -1.0).astype(np.float32)
        row = (np.arange(m, dtype=np.int64) // t).astype(np.int64)

        order = np.argsort(idx, kind="stable")
        return GatherProjSpec(
            dim_ambient=D,
            d=d,
            t=t,
            seed=int(seed),
            idx_sorted=idx[order],
            row_sorted=row[order],
            sign_sorted=sign[order],
        )


def project_flat(x: torch.Tensor, *, spec: GatherProjSpec) -> torch.Tensor:
    """Project a 1-D tensor x of length D into R^d using spec."""
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if int(x.numel()) != int(spec.dim_ambient):
        raise ValueError(f"x has dim {int(x.numel())}, expected {int(spec.dim_ambient)}")
    device = x.device
    y = torch.zeros((int(spec.d),), dtype=torch.float32, device=device)
    idx_t = torch.from_numpy(spec.idx_sorted).to(device=device, dtype=torch.int64)
    rows_t = torch.from_numpy(spec.row_sorted).to(device=device, dtype=torch.int64)
    signs_t = torch.from_numpy(spec.sign_sorted).to(device=device, dtype=torch.float32)
    vals = x.index_select(0, idx_t).float()
    y.index_add_(0, rows_t, vals * signs_t)
    y.mul_(1.0 / math.sqrt(int(spec.t)))
    return y


def project_module(module: nn.Module, *, spec: GatherProjSpec) -> torch.Tensor:
    """Project an nn.Module's flattened parameters into R^d without O(D) buffers.

    Scans the module parameter tensors and gathers only the O(d*t) required
    coordinates via index_select per-parameter, then accumulates into y.
    """
    device = next(module.parameters()).device
    y = torch.zeros((int(spec.d),), dtype=torch.float32, device=device)
    inv_sqrt_t = 1.0 / math.sqrt(int(spec.t))

    idx = spec.idx_sorted
    row = spec.row_sorted
    sign = spec.sign_sorted
    m = int(idx.shape[0])
    lo = 0
    offset = 0
    for p in module.parameters():
        flat = p.detach().reshape(-1)
        n = int(flat.numel())
        if lo >= m:
            break
        hi = int(np.searchsorted(idx, offset + n, side="left"))
        if lo < hi:
            local = (idx[lo:hi] - offset).astype(np.int64, copy=False)
            local_t = torch.from_numpy(local).to(device=device, dtype=torch.int64)
            rows_t = torch.from_numpy(row[lo:hi]).to(device=device, dtype=torch.int64)
            signs_t = torch.from_numpy(sign[lo:hi]).to(device=device, dtype=torch.float32)
            vals = flat.index_select(0, local_t).float()
            y.index_add_(0, rows_t, vals * signs_t)
            lo = hi
        offset += n

    y.mul_(inv_sqrt_t)
    return y
