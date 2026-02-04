from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
from enn.turbo.components.incumbent_selector import ScalarIncumbentSelector
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.config.tr_length_config import TRLengthConfig
from enn.turbo.turbo_trust_region import TurboTrustRegion
from enn.turbo.turbo_utils import generate_raasp_candidates

GeometryKind = Literal["box", "enn_ellipsoid", "enn_grad_ellipsoid"]
SamplerKind = Literal["full", "low_rank"]


class _LowRankFactor(NamedTuple):
    sqrt_alpha: float
    basis: np.ndarray
    sqrt_vals: np.ndarray


def _normalize_weights(weights: np.ndarray) -> np.ndarray | None:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return w / total


def _trace_normalize(cov: np.ndarray, dim: int) -> np.ndarray:
    trace = float(np.trace(cov))
    if not np.isfinite(trace) or trace <= 0.0:
        return np.eye(dim, dtype=float)
    return cov / trace * float(dim)


def _clip_and_rescale_eigs(eigvals: np.ndarray, *, dim: int, lam_min: float, lam_max: float) -> np.ndarray | None:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.clip(eigvals, lam_min, lam_max)
    total = float(np.sum(eigvals))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return eigvals * (float(dim) / total)


def _full_factor(cov: np.ndarray, *, dim: int, lam_min: float, lam_max: float, eps: float) -> np.ndarray | None:
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    eigvals = _clip_and_rescale_eigs(eigvals, dim=dim, lam_min=lam_min, lam_max=lam_max)
    if eigvals is None:
        return None
    return eigvecs * np.sqrt(eigvals + eps).reshape(1, -1)


def _low_rank_factor(
    centered: np.ndarray,
    weights: np.ndarray,
    *,
    dim: int,
    lam_min: float,
    lam_max: float,
    eps: float,
    rank_cap: int | None,
) -> _LowRankFactor | None:
    b = centered * np.sqrt(weights).reshape(-1, 1)
    try:
        _u, svals, vt = np.linalg.svd(b, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if svals.size == 0 or not np.all(np.isfinite(svals)):
        return None
    lam_all = np.square(svals)
    if not np.all(np.isfinite(lam_all)):
        return None
    r = int(min(dim, lam_all.size, centered.shape[0]))
    if rank_cap is not None:
        r = min(r, max(int(rank_cap), 0))
    if r <= 0:
        return None
    v = vt[:r].T
    lam = lam_all[:r]
    total = float(np.sum(lam))
    if not np.isfinite(total) or total <= 0.0:
        return None
    lam = lam / total * float(dim)
    lam = np.clip(lam, lam_min, lam_max)
    lam = _clip_and_rescale_eigs(lam, dim=dim, lam_min=lam_min, lam_max=lam_max)
    if lam is None:
        return None
    alpha0 = 1e-4
    trace_total = alpha0 * float(dim) + float(np.sum(lam))
    scale = float(dim) / trace_total
    alpha = alpha0 * scale
    lam = lam * scale
    sqrt_alpha = float(np.sqrt(alpha))
    sqrt_lam = np.sqrt(lam + eps)
    return _LowRankFactor(sqrt_alpha=sqrt_alpha, basis=v, sqrt_vals=sqrt_lam)


def _clip_to_unit_box(x_center: np.ndarray, step: np.ndarray) -> np.ndarray:
    num_candidates, num_dim = step.shape
    t = np.ones((num_candidates,), dtype=float)
    for j in range(num_dim):
        sj = step[:, j]
        if float(x_center[j]) < 0.0 or float(x_center[j]) > 1.0:
            raise ValueError("x_center must lie in [0, 1]^D")
        pos = sj > 0.0
        if np.any(pos):
            t[pos] = np.minimum(t[pos], (1.0 - x_center[j]) / sj[pos])
        neg = sj < 0.0
        if np.any(neg):
            t[neg] = np.minimum(t[neg], (0.0 - x_center[j]) / sj[neg])
    t = np.clip(t, 0.0, 1.0)
    return x_center.reshape(1, -1) + step * t.reshape(-1, 1)


def _add_sparse_axis(step: np.ndarray, z: np.ndarray, scale: float) -> None:
    if scale == 0.0:
        return
    nz_cols = np.where(np.any(z != 0.0, axis=0))[0]
    for j in nz_cols:
        rows = np.where(z[:, j] != 0.0)[0]
        if rows.size == 0:
            continue
        step[rows, j] += scale * z[rows, j]


def _apply_full_factor(z: np.ndarray, factor: np.ndarray) -> np.ndarray:
    num_candidates, num_dim = z.shape
    if factor.shape != (num_dim, num_dim):
        raise ValueError(f"full factor must be ({num_dim}, {num_dim}), got {factor.shape}")
    step = np.zeros((num_candidates, num_dim), dtype=float)
    nz_cols = np.where(np.any(z != 0.0, axis=0))[0]
    for j in nz_cols:
        rows = np.where(z[:, j] != 0.0)[0]
        if rows.size == 0:
            continue
        step[rows] += z[rows, j].reshape(-1, 1) * factor[:, j].reshape(1, -1)
    return step


@dataclass(frozen=True)
class EllipsoidTRConfig:
    length: TRLengthConfig = TRLengthConfig()
    fixed_length: float | None = None
    noise_aware: bool = False
    geometry: GeometryKind = "box"
    ellipsoid_sampler: SamplerKind | None = None
    ellipsoid_rank: int | None = None

    def __post_init__(self) -> None:
        if self.geometry not in (
            "box",
            "enn_ellipsoid",
            "enn_grad_ellipsoid",
        ):
            raise ValueError(f"Unknown geometry={self.geometry!r}")
        if self.geometry == "box":
            if self.ellipsoid_sampler is not None or self.ellipsoid_rank is not None:
                raise ValueError("ellipsoid_* options require non-box geometry")
            if self.fixed_length is not None:
                raise ValueError("fixed_length requires non-box geometry")
        if self.geometry == "enn_grad_ellipsoid":
            if self.ellipsoid_sampler is not None and self.ellipsoid_sampler != "low_rank":
                raise ValueError("enn_grad_ellipsoid requires ellipsoid_sampler='low_rank'")
        if self.ellipsoid_rank is not None and int(self.ellipsoid_rank) <= 0:
            raise ValueError(f"ellipsoid_rank must be > 0, got {self.ellipsoid_rank}")
        if self.fixed_length is not None and float(self.fixed_length) <= 0:
            raise ValueError(f"fixed_length must be > 0, got {self.fixed_length}")

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max

    def _resolved_sampler(self) -> SamplerKind:
        if self.geometry == "enn_grad_ellipsoid":
            return "low_rank"
        return "full" if self.ellipsoid_sampler is None else self.ellipsoid_sampler

    def build(
        self,
        *,
        num_dim: int,
        rng: Any,
        candidate_rv: CandidateRV | None = None,
    ) -> TurboTrustRegion:
        selector = ScalarIncumbentSelector(noise_aware=self.noise_aware)
        if self.geometry == "box":
            return TurboTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
            )
        sampler = self._resolved_sampler()
        candidate_rv = CandidateRV.SOBOL if candidate_rv is None else candidate_rv
        if self.geometry in ("enn_ellipsoid", "enn_grad_ellipsoid"):
            return ENNEllipsoidTrustRegion(
                config=self,
                num_dim=num_dim,
                incumbent_selector=selector,
                candidate_rv=candidate_rv,
                ellipsoid_sampler=sampler,
                ellipsoid_rank=self.ellipsoid_rank,
            )
        raise ValueError(f"Unknown geometry={self.geometry!r}")


@dataclass
class EllipsoidTrustRegionBase(TurboTrustRegion):
    candidate_rv: CandidateRV = CandidateRV.SOBOL
    ellipsoid_sampler: SamplerKind = "full"
    ellipsoid_rank: int | None = None
    uses_custom_candidate_gen: bool = field(default=True, init=False)
    _cov_factor: np.ndarray | None = field(default=None, init=False, repr=False)
    _low_rank: _LowRankFactor | None = field(default=None, init=False, repr=False)
    has_geometry: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._reset_geometry()
        self._apply_fixed_length()

    def _reset_geometry(self) -> None:
        self._cov_factor = np.eye(self.num_dim, dtype=float)
        self._low_rank = _LowRankFactor(
            sqrt_alpha=1.0,
            basis=np.zeros((self.num_dim, 0), dtype=float),
            sqrt_vals=np.zeros(0),
        )
        self.has_geometry = False

    def _fixed_length(self) -> float | None:
        return getattr(self.config, "fixed_length", None)

    def _apply_fixed_length(self) -> None:
        fixed_length = self._fixed_length()
        if fixed_length is not None:
            self.length = float(fixed_length)

    def restart(self, rng: Any | None = None) -> None:
        super().restart(rng=rng)
        self._reset_geometry()
        self._apply_fixed_length()

    def update(self, y_obs: np.ndarray | Any, y_incumbent: np.ndarray | Any) -> None:
        super().update(y_obs, y_incumbent)
        self._apply_fixed_length()

    def needs_restart(self) -> bool:
        if self._fixed_length() is not None:
            return False
        return super().needs_restart()

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        dx = np.asarray(delta_x, dtype=float)
        if dx.ndim != 2 or dx.shape[0] == 0:
            return
        if dx.shape[1] != self.num_dim:
            raise ValueError(f"delta_x has incompatible shape {dx.shape} for num_dim={self.num_dim}")
        w = _normalize_weights(np.asarray(weights, dtype=float))
        if w is None or w.shape[0] != dx.shape[0]:
            return
        mean = np.sum(w[:, None] * dx, axis=0)
        centered = dx - mean
        cov = centered.T @ (w[:, None] * centered)
        cov = _trace_normalize(cov, self.num_dim)
        self._update_from_cov(centered, w, cov)

    def _update_from_cov(
        self,
        centered: np.ndarray,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> None:
        lam_min = 1e-4
        lam_max = 1e4
        eps = 1e-6
        if self.ellipsoid_sampler == "full":
            factor = _full_factor(cov, dim=self.num_dim, lam_min=lam_min, lam_max=lam_max, eps=eps)
            if factor is None:
                return
            self._cov_factor = factor
            self.has_geometry = True
            return
        if self.ellipsoid_sampler != "low_rank":
            raise ValueError(f"Unknown ellipsoid_sampler: {self.ellipsoid_sampler!r}")
        rank_cap = int(self.ellipsoid_rank) if self.ellipsoid_rank is not None else None
        low_rank = _low_rank_factor(
            centered,
            weights,
            dim=self.num_dim,
            lam_min=lam_min,
            lam_max=lam_max,
            eps=eps,
            rank_cap=rank_cap,
        )
        if low_rank is None:
            return
        self._low_rank = low_rank
        self._cov_factor = low_rank.basis * low_rank.sqrt_vals.reshape(1, -1)
        self.has_geometry = True

    def generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        num_candidates: int,
        *,
        rng: Any,
        candidate_rv: CandidateRV | None = None,
        sobol_engine: Any | None = None,
        raasp_driver: Any | None = None,
        num_pert: int = 20,
    ) -> np.ndarray:
        _ = raasp_driver
        if lengthscales is not None:
            raise ValueError("lengthscales are not supported for ellipsoid trust regions")
        num_candidates = int(num_candidates)
        if num_candidates <= 0:
            raise ValueError(num_candidates)
        num_dim = int(self.num_dim)
        if x_center.shape != (num_dim,):
            raise ValueError((x_center.shape, num_dim))

        z_center = np.zeros(num_dim, dtype=float)
        lb = -0.5 * np.ones(num_dim, dtype=float)
        ub = 0.5 * np.ones(num_dim, dtype=float)
        rv = self.candidate_rv if candidate_rv is None else candidate_rv
        z = generate_raasp_candidates(
            z_center,
            lb,
            ub,
            num_candidates,
            rng=rng,
            candidate_rv=rv,
            sobol_engine=sobol_engine,
            num_pert=num_pert,
        )

        step = self._build_step(z, rng)
        step = step * float(self.length)
        candidates = _clip_to_unit_box(x_center, step)
        if candidates.shape != (num_candidates, num_dim):
            raise RuntimeError((candidates.shape, (num_candidates, num_dim)))
        return candidates

    def _build_step(self, z: np.ndarray, rng: Any) -> np.ndarray:
        if self.ellipsoid_sampler == "full":
            if self._cov_factor is None:
                raise RuntimeError("trust region missing full-rank geometry")
            return _apply_full_factor(z, self._cov_factor)
        if self.ellipsoid_sampler != "low_rank":
            raise ValueError(self.ellipsoid_sampler)
        if self._low_rank is None:
            raise RuntimeError("trust region missing low-rank geometry")
        low_rank = self._low_rank
        basis = np.asarray(low_rank.basis, dtype=float)
        sqrt_vals = np.asarray(low_rank.sqrt_vals, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != self.num_dim:
            raise RuntimeError(basis.shape)
        if sqrt_vals.ndim != 1 or basis.shape[1] != sqrt_vals.shape[0]:
            raise RuntimeError((basis.shape, sqrt_vals.shape))
        rank = int(sqrt_vals.shape[0])
        if rank == 0:
            step = np.zeros((z.shape[0], self.num_dim), dtype=float)
        else:
            coeff = rng.uniform(-0.5, 0.5, size=(z.shape[0], rank)) * sqrt_vals.reshape(1, -1)
            step = coeff @ basis.T
        _add_sparse_axis(step, z, float(low_rank.sqrt_alpha))
        return step


@dataclass
class ENNEllipsoidTrustRegion(EllipsoidTrustRegionBase):
    has_enn_geometry: bool = field(default=False, init=False)

    def set_geometry(self, delta_x: np.ndarray | Any, weights: np.ndarray | Any) -> None:
        super().set_geometry(delta_x, weights)
        self.has_enn_geometry = self.has_geometry

    def set_gradient_geometry(
        self,
        delta_x: np.ndarray | Any,
        delta_y: np.ndarray | Any,
        weights: np.ndarray | Any,
        *,
        eps_norm: float = 1e-12,
    ) -> None:
        dx = np.asarray(delta_x, dtype=float)
        if dx.ndim != 2 or dx.shape[0] == 0:
            return
        if dx.shape[1] != self.num_dim:
            raise ValueError(f"delta_x has incompatible shape {dx.shape} for num_dim={self.num_dim}")
        dy = np.asarray(delta_y, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if dy.shape[0] != dx.shape[0] or w.shape[0] != dx.shape[0]:
            raise ValueError((dy.shape, w.shape, dx.shape))
        w = _normalize_weights(w)
        if w is None:
            return
        norms = np.linalg.norm(dx, axis=1)
        scale = np.abs(dy) / np.maximum(norms, float(eps_norm))
        scale = np.where(np.isfinite(scale), scale, 0.0)
        if not np.any(scale > 0.0):
            return
        centered = dx * (np.sqrt(w) * scale).reshape(-1, 1)
        cov = centered.T @ centered
        cov = _trace_normalize(cov, self.num_dim)
        self._update_from_cov(centered, w, cov)
        self.has_enn_geometry = self.has_geometry
