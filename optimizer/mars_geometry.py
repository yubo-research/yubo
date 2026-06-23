from __future__ import annotations

from typing import NamedTuple

import numpy as np


class _LowRankFactor(NamedTuple):
    sqrt_alpha: float
    basis: np.ndarray
    sqrt_vals: np.ndarray


def _low_rank_factor_from_isotropic_spectrum(
    *,
    alpha_base: float,
    basis: np.ndarray,
    extra_eigvals: np.ndarray,
    dim: int,
    lam_min: float,
    lam_max: float,
    eps: float,
    rank_cap: int | None,
    kappa_max: float | None = None,
) -> _LowRankFactor | None:
    extra = _valid_extra_eigvals(extra_eigvals)
    if extra is None:
        return None
    basis_arr = _valid_basis(basis, dim=int(dim), extra_size=extra.size)
    order = np.argsort(extra)[::-1]
    extra = extra[order]
    basis_arr = basis_arr[:, order]
    rank = _resolved_rank(dim=int(dim), extra_size=extra.size, rank_cap=rank_cap)
    alpha0 = _alpha0(
        alpha_base=float(max(alpha_base, 0.0)),
        extra=extra,
        dim=int(dim),
        rank=rank,
        kappa_max=kappa_max,
    )
    alpha0 = float(np.clip(alpha0, lam_min, lam_max))
    if rank > 0:
        top_eigs = float(max(alpha_base, 0.0)) + extra[:rank]
        lam = np.maximum(top_eigs - alpha0, 0.0)
        basis_keep = basis_arr[:, :rank]
    else:
        lam = np.zeros((0,), dtype=float)
        basis_keep = np.zeros((int(dim), 0), dtype=float)
    total = alpha0 * float(dim) + float(np.sum(lam))
    if not np.isfinite(total) or total <= 0.0:
        return None
    scale = float(dim) / total
    return _LowRankFactor(
        sqrt_alpha=float(np.sqrt(max(alpha0 * scale, 0.0))),
        basis=basis_keep,
        sqrt_vals=np.sqrt(lam * scale + eps),
    )


def _valid_extra_eigvals(extra_eigvals: np.ndarray) -> np.ndarray | None:
    extra = np.asarray(extra_eigvals, dtype=float).reshape(-1)
    extra = np.maximum(extra, 0.0)
    if not np.all(np.isfinite(extra)):
        return None
    return extra


def _valid_basis(basis: np.ndarray, *, dim: int, extra_size: int) -> np.ndarray:
    basis_arr = np.asarray(basis, dtype=float)
    if basis_arr.ndim == 2 and basis_arr.shape == (dim, extra_size):
        if extra_size > 0 and not np.all(np.isfinite(basis_arr)):
            raise ValueError("basis contains non-finite values")
        return basis_arr
    if extra_size == 0:
        return np.zeros((dim, 0), dtype=float)
    raise ValueError((basis_arr.shape, extra_size, dim))


def _resolved_rank(*, dim: int, extra_size: int, rank_cap: int | None) -> int:
    max_rank = int(min(dim, extra_size))
    if rank_cap is None:
        return max_rank
    return min(max_rank, max(int(rank_cap), 0))


def _alpha0(
    *,
    alpha_base: float,
    extra: np.ndarray,
    dim: int,
    rank: int,
    kappa_max: float | None,
) -> float:
    if rank < min(dim, extra.size):
        alpha = alpha_base + float(np.sum(extra[rank:])) / float(max(dim - rank, 1))
    elif rank < dim:
        alpha = alpha_base
    elif extra.size > 0:
        alpha = alpha_base + float(np.min(extra))
    else:
        alpha = alpha_base
    if kappa_max is not None and float(kappa_max) >= 1.0:
        max_full_eig = alpha_base + (float(extra[0]) if extra.size > 0 else 0.0)
        alpha = max(alpha, max_full_eig / float(kappa_max))
    return alpha
