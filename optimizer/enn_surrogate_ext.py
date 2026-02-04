from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.components.enn_surrogate import ENNSurrogate


def _ensure_single_point(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2 or x.shape != (1, dim):
        raise ValueError(f"x must be a single point with {dim} dims, got {x.shape}")
    return x


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return np.full((w.size,), 1.0 / float(w.size), dtype=float)
    return w / total


@dataclass(frozen=True)
class LocalGeometryValues:
    delta_x: np.ndarray
    y_neighbors: np.ndarray
    weights: np.ndarray


def _require_scalar_y_obs(y_obs: np.ndarray, *, context: str) -> np.ndarray:
    y = np.asarray(y_obs, dtype=float)
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(f"{context} requires scalar y_obs (shape (n,) or (n,1)).")
        return y[:, 0]
    if y.ndim == 1:
        return y
    raise ValueError(f"{context} requires scalar y_obs (shape (n,) or (n,1)).")


class GeometryENNSurrogate(ENNSurrogate):
    def _require_fitted(self) -> tuple[Any, Any]:
        if self._enn is None or self._params is None:
            raise RuntimeError("GeometryENNSurrogate requires a fitted ENN surrogate.")
        return self._enn, self._params

    def _require_enn_internals(
        self,
        *,
        need_train_y: bool = False,
        need_num_outputs: bool = False,
    ) -> tuple[Any, Any]:
        enn, params = self._require_fitted()
        missing = []
        if not hasattr(enn, "_compute_posterior_internals"):
            missing.append("_compute_posterior_internals")
        if not hasattr(enn, "train_x"):
            missing.append("train_x")
        if need_train_y and not hasattr(enn, "train_y"):
            missing.append("train_y")
        if need_num_outputs and not hasattr(enn, "num_outputs"):
            missing.append("num_outputs")
        if not hasattr(params, "k_num_neighbors"):
            missing.append("k_num_neighbors")
        if missing:
            raise RuntimeError(f"ENN internals missing required attributes: {', '.join(missing)}. Please pin `ennbo` to a compatible version.")
        return enn, params

    def local_geometry(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        enn, params = self._require_enn_internals()
        params = self._params if params is None else params
        x = _ensure_single_point(x, enn.train_x.shape[1])
        if len(enn) == 0 or int(params.k_num_neighbors) <= 0:
            return np.zeros((0, x.shape[1]), dtype=float), np.zeros((0,), dtype=float)
        if exclude_nearest and len(enn) <= 1:
            return np.zeros((0, x.shape[1]), dtype=float), np.zeros((0,), dtype=float)
        internals = enn._compute_posterior_internals(x, params, PosteriorFlags(exclude_nearest=exclude_nearest, observation_noise=False))
        idx = internals.idx
        if idx.shape[0] != 1 or idx.shape[1] == 0:
            return np.zeros((0, x.shape[1]), dtype=float), np.zeros((0,), dtype=float)
        weights = _normalize_weights(internals.w_normalized[0, :, 0])
        x_neighbors = enn.train_x[idx[0]].copy()
        return x_neighbors - x[0], weights

    def local_geometry_values(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
    ) -> LocalGeometryValues:
        enn, params = self._require_enn_internals(need_train_y=True, need_num_outputs=True)
        params = self._params if params is None else params
        x = _ensure_single_point(x, enn.train_x.shape[1])
        if len(enn) == 0 or int(params.k_num_neighbors) <= 0:
            empty_dx = np.zeros((0, x.shape[1]), dtype=float)
            return LocalGeometryValues(
                delta_x=empty_dx,
                y_neighbors=np.zeros((0, enn.num_outputs), dtype=float),
                weights=np.zeros((0,), dtype=float),
            )
        if exclude_nearest and len(enn) <= 1:
            empty_dx = np.zeros((0, x.shape[1]), dtype=float)
            return LocalGeometryValues(
                delta_x=empty_dx,
                y_neighbors=np.zeros((0, enn.num_outputs), dtype=float),
                weights=np.zeros((0,), dtype=float),
            )
        internals = enn._compute_posterior_internals(x, params, PosteriorFlags(exclude_nearest=exclude_nearest, observation_noise=False))
        idx = internals.idx
        if idx.shape[0] != 1 or idx.shape[1] == 0:
            empty_dx = np.zeros((0, x.shape[1]), dtype=float)
            return LocalGeometryValues(
                delta_x=empty_dx,
                y_neighbors=np.zeros((0, enn.num_outputs), dtype=float),
                weights=np.zeros((0,), dtype=float),
            )
        weights = _normalize_weights(internals.w_normalized[0, :, 0])
        x_neighbors = enn.train_x[idx[0]].copy()
        y_neighbors = enn.train_y[idx[0]].copy()
        return LocalGeometryValues(
            delta_x=x_neighbors - x[0],
            y_neighbors=y_neighbors,
            weights=weights,
        )

    def _update_enn_ellipsoid(self, tr_state: Any, x_center: np.ndarray) -> None:
        delta_x, weights = self.local_geometry(x_center, params=self._params, exclude_nearest=True)
        tr_state.set_geometry(delta_x=delta_x, weights=weights)

    def _update_grad_ellipsoid(
        self,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int,
    ) -> None:
        if not hasattr(self._enn, "num_outputs"):
            raise RuntimeError("ENN internals missing num_outputs; please pin `ennbo` to a compatible version.")
        if int(self._enn.num_outputs) != 1:
            raise ValueError("enn_grad_ellipsoid requires scalar ENN outputs.")
        y_flat = _require_scalar_y_obs(y_obs, context="enn_grad_ellipsoid")
        geom = self.local_geometry_values(x_center, params=self._params, exclude_nearest=True)
        if geom.y_neighbors.size == 0:
            return
        center_val = float(y_flat[int(incumbent_idx)])
        neighbor_vals = np.asarray(geom.y_neighbors[:, 0], dtype=float).reshape(-1)
        delta_y = neighbor_vals - center_val
        tr_state.set_gradient_geometry(delta_x=geom.delta_x, delta_y=delta_y, weights=geom.weights)

    def update_trust_region(
        self,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int | None,
        rng: Any,
    ) -> None:
        from optimizer.ellipsoid_trust_region import ENNEllipsoidTrustRegion

        _ = rng
        if incumbent_idx is None or not isinstance(tr_state, ENNEllipsoidTrustRegion) or self._enn is None or self._params is None:
            return
        geometry = getattr(tr_state.config, "geometry", "box")
        if geometry == "enn_ellipsoid":
            self._update_enn_ellipsoid(tr_state, x_center)
            return
        if geometry == "enn_grad_ellipsoid":
            self._update_grad_ellipsoid(tr_state, x_center, y_obs=y_obs, incumbent_idx=int(incumbent_idx))
            return
