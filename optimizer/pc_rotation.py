"""
LABCAT-style principal-component (PC) rotation for trust regions.

Implements the weighted principal component rotation from:
  https://github.com/esl-sun/LABCAT
  (Visser, van Daalen, Schoeman. LABCAT: Locally Adaptive Bayesian Optimisation
   using Principal-Component-Aligned Trust Regions. Swarm Evol. Comput.)

The rotation aligns the trust region with the weighted principal components
of the observed data, where better points (higher objective for maximization)
receive higher weight. This enables the trust region to expand along
directions of local separability (e.g. valleys) and contract along
less informative directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# -----------------------------------------------------------------------------
# LABCAT paper citation (for in-code reference)
# -----------------------------------------------------------------------------
LABCAT_CITATION = (
    "https://arxiv.org/abs/2311.11328 — "
    "Visser, van Daalen, Schoeman. LABCAT: Locally adaptive Bayesian "
    "optimization using principal-component-aligned trust regions. "
    "arXiv:2311.11328, 2023. Sec. 3 (Principal-component-based rotation)."
)

PCRotationMode = Literal["full", "low_rank"]


@dataclass(frozen=True)
class PCRotationResult:
    """Result of LABCAT-style weighted PCA for trust region rotation.

    Attributes:
        center: Center point (incumbent) used for centering, shape (d,).
        basis: Orthonormal basis columns = principal directions, shape (d, r).
               r = min(n, d) for full, r = min(k, n, d) for low_rank.
        singular_values: Singular values (radii) for each principal direction.
        has_rotation: True if a valid rotation was computed.
    """

    center: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    has_rotation: bool

    def to_rotated(self, x: np.ndarray) -> np.ndarray:
        """Project points into rotated (PC) space: x' = basis.T @ (x - center)."""
        if not self.has_rotation:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            delta = x - self.center
            return (self.basis.T @ delta).reshape(-1)
        delta = x - self.center.reshape(1, -1)
        return delta @ self.basis

    def from_rotated(self, x_rot: np.ndarray) -> np.ndarray:
        """Map points from rotated space back to original: x = center + basis @ x'."""
        if not self.has_rotation:
            return np.asarray(x_rot, dtype=float)
        x_rot = np.asarray(x_rot, dtype=float)
        if x_rot.ndim == 1:
            return (self.center + self.basis @ x_rot).reshape(-1)
        return self.center.reshape(1, -1) + (x_rot @ self.basis.T)


def _labcat_weights_maximization(y: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """LABCAT-style weights for maximization: w_i ∝ (y_i - y_min).

    Better points (higher y) get higher weight. Normalized to sum to 1.
    For minimization, LABCAT uses w_i = 1 - ỹ_i; we use ỹ_i for maximization.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = y_max - y_min
    if not np.isfinite(span) or span <= 0.0:
        return np.ones(y.size, dtype=float) / max(1, y.size)
    # ỹ = (y - y_min) / span ∈ [0, 1]; better points → higher ỹ
    y_norm = (y - y_min) / span
    w = np.maximum(y_norm, eps)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(y.size, dtype=float) / max(1, y.size)
    return w / total


def compute_labcat_weighted_pca(
    x_center: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    *,
    maximize: bool = True,
    mode: PCRotationMode = "full",
    rank: int | None = None,
    min_obs: int = 2,
    eps_svd: float = 1e-10,
) -> PCRotationResult:
    """Compute LABCAT-style weighted PCA for trust region rotation.

    Centers data on x_center, applies sample weights biased toward better
    objective values, and computes principal directions via thin SVD.

    Args:
        x_center: Incumbent / center point, shape (d,).
        x_obs: Observed inputs, shape (n, d).
        y_obs: Observed objective values, shape (n,) or (n, 1).
        maximize: If True, higher y = better; if False, lower y = better.
        mode: "full" = use all min(n,d) PCs; "low_rank" = use top rank PCs.
        rank: For low_rank mode, max number of PCs (e.g. 50–100).
        min_obs: Minimum observations required to compute rotation.
        eps_svd: Small constant for numerical stability in SVD.

    Returns:
        PCRotationResult with center, basis, singular_values, has_rotation.
    """
    x_center = np.asarray(x_center, dtype=float).reshape(-1)
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)

    if x_obs.ndim != 2 or x_obs.shape[0] == 0:
        return PCRotationResult(
            center=x_center,
            basis=np.eye(x_center.size, dtype=float),
            singular_values=np.ones(x_center.size, dtype=float),
            has_rotation=False,
        )
    if x_obs.shape[1] != x_center.size:
        raise ValueError(f"x_obs shape {x_obs.shape} incompatible with x_center dim {x_center.size}")
    if y_obs.shape[0] != x_obs.shape[0]:
        raise ValueError(f"y_obs length {y_obs.shape[0]} must match x_obs rows {x_obs.shape[0]}")
    if x_obs.shape[0] < min_obs:
        return PCRotationResult(
            center=x_center,
            basis=np.eye(x_center.size, dtype=float),
            singular_values=np.ones(x_center.size, dtype=float),
            has_rotation=False,
        )

    # Center on incumbent
    centered = x_obs - x_center.reshape(1, -1)

    # LABCAT weights: better points get higher weight
    if maximize:
        weights = _labcat_weights_maximization(y_obs)
    else:
        # Minimization: w_i ∝ (y_max - y_i)
        weights = _labcat_weights_maximization(-y_obs)

    # Weighted data: X_w = centered * sqrt(w) per row
    sqrt_w = np.sqrt(np.maximum(weights, 1e-12))
    x_weighted = centered * sqrt_w.reshape(-1, 1)

    try:
        u, s, vh = np.linalg.svd(x_weighted, full_matrices=False)
    except np.linalg.LinAlgError:
        return PCRotationResult(
            center=x_center,
            basis=np.eye(x_center.size, dtype=float),
            singular_values=np.ones(x_center.size, dtype=float),
            has_rotation=False,
        )

    if s.size == 0 or not np.all(np.isfinite(s)):
        return PCRotationResult(
            center=x_center,
            basis=np.eye(x_center.size, dtype=float),
            singular_values=np.ones(x_center.size, dtype=float),
            has_rotation=False,
        )

    # V from SVD: columns = principal directions in original space
    # vh is (r, d), so V = vh.T is (d, r)
    v = vh.T
    s_safe = np.maximum(s, eps_svd)

    if mode == "low_rank" and rank is not None:
        k = min(int(rank), v.shape[1], s.size)
        if k <= 0:
            return PCRotationResult(
                center=x_center,
                basis=np.eye(x_center.size, dtype=float),
                singular_values=np.ones(x_center.size, dtype=float),
                has_rotation=False,
            )
        basis = v[:, :k].copy()
        singular_values = s_safe[:k].copy()
    else:
        basis = v.copy()
        singular_values = s_safe.copy()

    return PCRotationResult(
        center=x_center.copy(),
        basis=basis,
        singular_values=singular_values,
        has_rotation=True,
    )
