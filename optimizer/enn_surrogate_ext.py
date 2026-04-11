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
    z = np.sum(u)
    if not np.isfinite(z) or z <= 0.0:
        return None

    y_neighbors = train_y[idx]
    if y_neighbors.ndim == 2:
        y_neighbors = y_neighbors[:, 0]
    y_neighbors = np.asarray(y_neighbors, dtype=float).reshape(-1)

    du_ddist2 = -epi_scale / (v**2)
    ddist2_dx = 2.0 * delta
    dz_dx = np.sum(du_ddist2[:, np.newaxis] * ddist2_dx, axis=0)
    dw_dx = (du_ddist2[:, np.newaxis] * ddist2_dx * z - u[:, np.newaxis] * dz_dx) / (z**2)

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


@dataclass(frozen=True)
class LocalPosterior:
    x: np.ndarray
    idx: np.ndarray
    weights: np.ndarray
    x_neighbors: np.ndarray
    y_neighbors: np.ndarray | None = None

    @property
    def delta_x(self) -> np.ndarray:
        return self.x_neighbors - self.x


def _empty_local_geometry(num_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, num_dim), dtype=float), np.zeros((0,), dtype=float)


def _empty_local_geometry_values(num_dim: int, num_outputs: int) -> LocalGeometryValues:
    return LocalGeometryValues(
        delta_x=np.zeros((0, num_dim), dtype=float),
        y_neighbors=np.zeros((0, num_outputs), dtype=float),
        weights=np.zeros((0,), dtype=float),
    )


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


def _needs_gradient_signal(tr_state: Any) -> bool:
    needs_gradient_fn = getattr(tr_state, "needs_gradient_signal", None)
    return bool(needs_gradient_fn()) if callable(needs_gradient_fn) else False


def _needs_local_geometry(tr_state: Any) -> bool:
    needs_local_fn = getattr(tr_state, "needs_local_geometry", None)
    return bool(needs_local_fn()) if callable(needs_local_fn) else True


class GeometryENNSurrogate(ENNSurrogate):
    def _maybe_update_true_ellipsoid_rho(
        self,
        *,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int,
    ) -> None:
        geometry = str(getattr(getattr(tr_state, "config", None), "geometry", getattr(tr_state, "geometry", ""))).strip().lower()
        if geometry not in {"enn_ellip", "grad_ellip"}:
            return
        observe_transition = getattr(tr_state, "observe_incumbent_transition", None)
        if callable(observe_transition):
            y_flat = _require_scalar_y_obs(y_obs, context="enn_ellip")
            if not 0 <= int(incumbent_idx) < y_flat.shape[0]:
                return
            curr_val = float(y_flat[int(incumbent_idx)])
            observe_transition(
                x_center=np.asarray(x_center, dtype=float).reshape(-1),
                y_value=curr_val,
                predict_delta=lambda prev_x, curr_x: _predict_pair_delta(self.predict, prev_x, curr_x),
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

    def _local_posterior(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
        need_train_y: bool = False,
        need_num_outputs: bool = False,
    ) -> tuple[Any, Any, LocalPosterior | None]:
        enn, default_params = self._require_enn_internals(
            need_train_y=need_train_y,
            need_num_outputs=need_num_outputs,
        )
        params = default_params if params is None else params
        x = _ensure_single_point(x, enn.train_x.shape[1])
        if len(enn) == 0 or int(params.k_num_neighbors) <= 0:
            return enn, params, None
        if exclude_nearest and len(enn) <= 1:
            return enn, params, None
        internals = enn._compute_posterior_internals(
            x,
            params,
            PosteriorFlags(exclude_nearest=exclude_nearest, observation_noise=False),
        )
        idx = np.asarray(internals.idx, dtype=int)
        if idx.shape[0] != 1 or idx.shape[1] == 0:
            return enn, params, None
        row_idx = idx[0]
        y_neighbors = enn.train_y[row_idx].copy() if need_train_y else None
        return (
            enn,
            params,
            LocalPosterior(
                x=x[0].copy(),
                idx=row_idx,
                weights=_normalize_weights(internals.w_normalized[0, :, 0]),
                x_neighbors=enn.train_x[row_idx].copy(),
                y_neighbors=y_neighbors,
            ),
        )

    def local_geometry(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        enn, _, local = self._local_posterior(
            x,
            params=params,
            exclude_nearest=exclude_nearest,
        )
        if local is None:
            return _empty_local_geometry(enn.train_x.shape[1])
        return local.delta_x, local.weights

    def gradient_mu(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
    ) -> np.ndarray | None:
        enn, params, local = self._local_posterior(
            x,
            params=params,
            exclude_nearest=exclude_nearest,
            need_train_y=True,
        )
        if local is None:
            return None
        return _gradient_mu_impl(x=local.x, idx=local.idx, enn=enn, params=params)

    def local_geometry_values(
        self,
        x: np.ndarray,
        *,
        params: Any | None = None,
        exclude_nearest: bool = True,
    ) -> LocalGeometryValues:
        enn, _, local = self._local_posterior(
            x,
            params=params,
            exclude_nearest=exclude_nearest,
            need_train_y=True,
            need_num_outputs=True,
        )
        if local is None:
            return _empty_local_geometry_values(enn.train_x.shape[1], int(enn.num_outputs))
        return LocalGeometryValues(
            delta_x=local.delta_x,
            y_neighbors=np.asarray(local.y_neighbors, dtype=float),
            weights=local.weights,
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

        if _needs_gradient_signal(tr_state):
            _observe_grad_metric_geometry(self, tr_state, x_center, y_obs, int(incumbent_idx))
        elif _needs_local_geometry(tr_state):
            _observe_local_metric_geometry(self, tr_state, x_center)

        _maybe_observe_pc_rotation(self, tr_state, x_center, y_obs)

        if callable(getattr(tr_state, "observe_incumbent_transition", None)):
            self._maybe_update_true_ellipsoid_rho(
                tr_state=tr_state,
                x_center=x_center,
                y_obs=y_obs,
                incumbent_idx=int(incumbent_idx),
            )


def _maybe_observe_pc_rotation(enn_surrogate, tr_state: Any, x_center: np.ndarray, y_obs: Any) -> None:
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


def _observe_grad_metric_geometry(
    enn_surrogate,
    tr_state: Any,
    x_center: np.ndarray,
    y_obs: np.ndarray,
    incumbent_idx: int,
) -> None:
    enn, params, local = enn_surrogate._local_posterior(
        x_center,
        params=enn_surrogate._params,
        exclude_nearest=True,
        need_train_y=True,
        need_num_outputs=True,
    )
    if not hasattr(enn, "num_outputs"):
        raise RuntimeError("ENN internals missing num_outputs; please pin `enn` to a compatible version.")
    if int(enn.num_outputs) != 1:
        raise ValueError("grad_metr requires scalar ENN outputs.")
    observe_local = getattr(tr_state, "observe_local_geometry", None)
    grad = None if local is None else _gradient_mu_impl(x=local.x, idx=local.idx, enn=enn, params=params)
    if grad is not None and np.any(np.isfinite(grad)):
        observe_local(grad=grad)
        return
    if local is not None and local.y_neighbors is not None and local.y_neighbors.size > 0:
        y_flat = _require_scalar_y_obs(y_obs, context="grad_metr")
        center_val = float(y_flat[int(incumbent_idx)])
        neighbor_vals = np.asarray(local.y_neighbors[:, 0], dtype=float).reshape(-1)
        observe_local(
            delta_x=local.delta_x,
            weights=local.weights,
            delta_y=neighbor_vals - center_val,
        )


def _observe_local_metric_geometry(
    enn_surrogate,
    tr_state: Any,
    x_center: np.ndarray,
) -> None:
    enn, _, local = enn_surrogate._local_posterior(
        x_center,
        params=enn_surrogate._params,
        exclude_nearest=True,
    )
    if local is None:
        delta_x, weights = _empty_local_geometry(enn.train_x.shape[1])
    else:
        delta_x, weights = local.delta_x, local.weights
    observe_local = getattr(tr_state, "observe_local_geometry", None)
    observe_local(delta_x=delta_x, weights=weights)
