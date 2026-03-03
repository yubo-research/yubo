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


def _gradient_mu_impl(
    x: np.ndarray,
    idx: np.ndarray,
    enn: Any,
    params: Any,
) -> np.ndarray | None:
    """Analytic gradient of μ(x) = Σ w_j(x) y_j for EpistemicNearestNeighbors."""
    eps_var = getattr(enn, "_EPS_VAR", 1e-9)
    scale_x = getattr(enn, "_scale_x", False)
    x_scale = getattr(enn, "_x_scale", np.ones((1, x.shape[0]), dtype=float))
    train_x = enn.train_x
    train_y = enn.train_y

    x_neighbors = train_x[idx]
    if scale_x:
        x_q = x / x_scale.reshape(-1)
        x_n = x_neighbors / x_scale.reshape(-1)
    else:
        x_q = np.asarray(x, dtype=float)
        x_n = np.asarray(x_neighbors, dtype=float)

    delta = x_q - x_n
    dist2s = np.sum(delta**2, axis=1)

    epi_scale = float(getattr(params, "epistemic_variance_scale", 1.0))
    var_ale = float(getattr(params, "aleatoric_variance_scale", 0.0))
    yvar = getattr(enn, "_train_yvar", None)
    if yvar is not None:
        var_ale = var_ale + yvar[idx] / (getattr(enn, "_y_scale", 1.0) ** 2)
    if np.isscalar(var_ale):
        var_ale = np.full(idx.shape[0], var_ale, dtype=float)
    else:
        var_ale = np.asarray(var_ale, dtype=float).reshape(-1)

    v = eps_var + epi_scale * dist2s + var_ale
    u = 1.0 / v
    Z = np.sum(u)
    if not np.isfinite(Z) or Z <= 0.0:
        return None

    y_neighbors = train_y[idx]
    if y_neighbors.ndim == 2:
        y_neighbors = y_neighbors[:, 0]
    y_neighbors = np.asarray(y_neighbors, dtype=float).reshape(-1)

    du_ddist2 = -epi_scale / (v**2)
    ddist2_dx = 2.0 * delta
    dZ_dx = np.sum(du_ddist2[:, np.newaxis] * ddist2_dx, axis=0)
    dw_dx = (du_ddist2[:, np.newaxis] * ddist2_dx * Z - u[:, np.newaxis] * dZ_dx) / (Z**2)

    grad_scaled = np.sum(dw_dx * y_neighbors[:, np.newaxis], axis=0)
    if scale_x:
        grad = grad_scaled / x_scale.reshape(-1)
    else:
        grad = grad_scaled
    return np.asarray(grad, dtype=float)


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


def _predict_pair_delta(predict_fn, prev_x: np.ndarray, curr_x: np.ndarray) -> float | None:
    pair = np.vstack(
        [
            np.asarray(prev_x, dtype=float).reshape(1, -1),
            np.asarray(curr_x, dtype=float).reshape(1, -1),
        ]
    )
    mu = np.asarray(predict_fn(pair).mu, dtype=float).reshape(-1)
    if mu.shape[0] >= 2 and np.all(np.isfinite(mu[:2])):
        return float(mu[1] - mu[0])
    return None


def _apply_acceptance_ratio_update(
    *,
    tr_state: Any,
    prev_x: np.ndarray,
    curr_x: np.ndarray,
    pred: float,
    act: float,
) -> None:
    cov = tr_state._covariance_matrix()
    delta = curr_x - np.asarray(prev_x, dtype=float).reshape(-1)
    solved = np.linalg.solve(cov, delta.reshape(-1, 1)).reshape(-1)
    dist = float(np.sqrt(max(0.0, np.dot(delta, solved))))
    length = float(getattr(tr_state, "length", 1.0))
    tol = float(getattr(tr_state.config, "boundary_tol", 0.1))
    boundary_hit = dist >= max(0.0, (1.0 - tol) * length)
    tr_state.set_acceptance_ratio(pred=pred, act=act, boundary_hit=boundary_hit)


class GeometryENNSurrogate(ENNSurrogate):
    def _maybe_update_true_ellipsoid_rho(
        self,
        *,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int,
    ) -> None:
        if hasattr(tr_state, "observe_incumbent_transition"):
            y_flat = _require_scalar_y_obs(y_obs, context="enn_true_ellipsoid")
            if int(incumbent_idx) < 0 or int(incumbent_idx) >= y_flat.shape[0]:
                return
            curr_val = float(y_flat[int(incumbent_idx)])

            tr_state.observe_incumbent_transition(
                x_center=np.asarray(x_center, dtype=float).reshape(-1),
                y_value=curr_val,
                predict_delta=lambda prev_x, curr_x: _predict_pair_delta(self.predict, prev_x, curr_x),
            )
            return

        if not hasattr(tr_state, "set_acceptance_ratio") or not hasattr(tr_state, "record_incumbent_transition"):
            return
        y_flat = _require_scalar_y_obs(y_obs, context="enn_true_ellipsoid")
        if int(incumbent_idx) < 0 or int(incumbent_idx) >= y_flat.shape[0]:
            return
        curr_val = float(y_flat[int(incumbent_idx)])
        curr_x = np.asarray(x_center, dtype=float).reshape(-1)
        prev_pair = tr_state.record_incumbent_transition(x_center=curr_x, y_value=curr_val)
        if prev_pair is None:
            return
        prev_val, prev_x = prev_pair
        act = float(curr_val - float(prev_val))
        pred = _predict_pair_delta(self.predict, np.asarray(prev_x, dtype=float).reshape(-1), curr_x)
        eps = 1e-12
        if pred is None or (not np.isfinite(pred)) or abs(float(pred)) < eps:
            return
        _apply_acceptance_ratio_update(
            tr_state=tr_state,
            prev_x=np.asarray(prev_x, dtype=float).reshape(-1),
            curr_x=curr_x,
            pred=float(pred),
            act=act,
        )

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
            raise RuntimeError(f"ENN internals missing required attributes: {', '.join(missing)}. Please pin `enn` to a compatible version.")
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
        internals = enn._compute_posterior_internals(
            x,
            params,
            PosteriorFlags(exclude_nearest=exclude_nearest, observation_noise=False),
        )
        idx = internals.idx
        if idx.shape[0] != 1 or idx.shape[1] == 0:
            return np.zeros((0, x.shape[1]), dtype=float), np.zeros((0,), dtype=float)
        weights = _normalize_weights(internals.w_normalized[0, :, 0])
        x_neighbors = enn.train_x[idx[0]].copy()
        return x_neighbors - x[0], weights

    def gradient_mu(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
    ) -> np.ndarray | None:
        """Analytic gradient of posterior mean μ(x) w.r.t. x. Returns None if unavailable."""
        enn, params = self._require_enn_internals(need_train_y=True, need_num_outputs=True)
        params = self._params if params is None else params
        x = _ensure_single_point(x, enn.train_x.shape[1])
        if len(enn) == 0 or int(params.k_num_neighbors) <= 0:
            return None
        if exclude_nearest and len(enn) <= 1:
            return None
        internals = enn._compute_posterior_internals(
            x,
            params,
            PosteriorFlags(exclude_nearest=exclude_nearest, observation_noise=False),
        )
        idx = internals.idx
        if idx.shape[0] != 1 or idx.shape[1] == 0:
            return None
        return _gradient_mu_impl(x=x[0], idx=idx[0], enn=enn, params=params)

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
        internals = enn._compute_posterior_internals(
            x,
            params,
            PosteriorFlags(exclude_nearest=exclude_nearest, observation_noise=False),
        )
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

    def update_trust_region(
        self,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int | None,
        rng: Any,
    ) -> None:
        _ = rng
        if incumbent_idx is None or self._enn is None or self._params is None:
            return
        observe_local = getattr(tr_state, "observe_local_geometry", None)
        if not callable(observe_local):
            return

        needs_gradient = False
        needs_gradient_fn = getattr(tr_state, "needs_gradient_signal", None)
        if callable(needs_gradient_fn):
            needs_gradient = bool(needs_gradient_fn())

        if needs_gradient:
            if not hasattr(self._enn, "num_outputs"):
                raise RuntimeError("ENN internals missing num_outputs; please pin `enn` to a compatible version.")
            if int(self._enn.num_outputs) != 1:
                raise ValueError("enn_grad_metric_shaped requires scalar ENN outputs.")
            grad = self.gradient_mu(x_center, params=self._params, exclude_nearest=True)
            if grad is not None and np.any(np.isfinite(grad)):
                observe_local(grad=grad)
            else:
                y_flat = _require_scalar_y_obs(y_obs, context="enn_grad_metric_shaped")
                geom = self.local_geometry_values(x_center, params=self._params, exclude_nearest=True)
                if geom.y_neighbors.size > 0:
                    center_val = float(y_flat[int(incumbent_idx)])
                    neighbor_vals = np.asarray(geom.y_neighbors[:, 0], dtype=float).reshape(-1)
                    delta_y = neighbor_vals - center_val
                    observe_local(
                        delta_x=geom.delta_x,
                        weights=geom.weights,
                        delta_y=delta_y,
                    )
        else:
            delta_x, weights = self.local_geometry(x_center, params=self._params, exclude_nearest=True)
            observe_local(delta_x=delta_x, weights=weights)

        _maybe_observe_pc_rotation(self, tr_state, x_center, y_obs)

        if callable(getattr(tr_state, "observe_incumbent_transition", None)):
            self._maybe_update_true_ellipsoid_rho(
                tr_state=tr_state,
                x_center=x_center,
                y_obs=y_obs,
                incumbent_idx=int(incumbent_idx),
            )


def _maybe_observe_pc_rotation(enn_surrogate, tr_state: Any, x_center: np.ndarray, y_obs: Any) -> None:
    """LABCAT-style PC rotation when enabled (see optimizer.pc_rotation)."""
    observe_pc = getattr(tr_state, "observe_pc_rotation_geometry", None)
    if not callable(observe_pc) or getattr(tr_state.config, "pc_rotation_mode", None) is None:
        return
    x_obs = np.asarray(enn_surrogate._enn.train_x, dtype=float)
    y_obs_arr = np.asarray(y_obs, dtype=float)
    if y_obs_arr.ndim == 2 and y_obs_arr.shape[1] > 1:
        scalarize = getattr(tr_state, "scalarize", None)
        y_scalar = scalarize(y_obs_arr, clip=False) if callable(scalarize) else None
        y_scalar = np.asarray(y_scalar, dtype=float).reshape(-1) if y_scalar is not None else y_obs_arr[:, 0]
    else:
        y_scalar = y_obs_arr.reshape(-1)
    if x_obs.shape[0] == y_scalar.shape[0] and x_obs.shape[0] >= 2:
        observe_pc(x_center=x_center, x_obs=x_obs, y_obs=y_scalar, maximize=True)
