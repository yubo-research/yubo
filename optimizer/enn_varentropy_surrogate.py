from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.python_fallback.components import PosteriorResult, SurrogateResult

from .enn_index_driver import parse_enn_index_driver
from .enn_varentropy_config import ENNVarentropySurrogateConfig


@dataclass
class _FittedENNVarentropyModel:
    enn_model: EpistemicNearestNeighbors
    train_x: np.ndarray
    train_y: np.ndarray
    train_yvar: np.ndarray | None
    y_scale: float


class ENNVarentropySurrogate:
    def __init__(self, config: ENNVarentropySurrogateConfig) -> None:
        self._config = config
        self._model: _FittedENNVarentropyModel | None = None

    @property
    def lengthscales(self) -> np.ndarray | None:
        return None

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: np.random.Generator | None = None,
    ) -> SurrogateResult:
        del num_steps, rng
        x = np.asarray(x_obs, dtype=float)
        y = _coerce_scalar_y(y_obs)
        yv = _coerce_yvar(y_var, y.shape[0])
        if _can_append(self._model, x, y, yv):
            assert self._model is not None
            _append_training_rows(self._model, x, y, yv)
            return SurrogateResult(model=self._model, lengthscales=None)
        enn_model = EpistemicNearestNeighbors(
            x,
            y.reshape(-1, 1),
            None if yv is None else yv.reshape(-1, 1),
            scale_x=bool(self._config.scale_x),
            index_driver=parse_enn_index_driver(self._config.index_driver),
        )
        self._model = _FittedENNVarentropyModel(
            enn_model=enn_model,
            train_x=x.copy(),
            train_y=y.copy(),
            train_yvar=None if yv is None else yv.copy(),
            y_scale=_y_scale(y),
        )
        return SurrogateResult(model=self._model, lengthscales=None)

    def get_incumbent_candidate_indices(self, y_obs: np.ndarray) -> np.ndarray:
        y = _coerce_scalar_y(y_obs)
        return np.arange(y.shape[0], dtype=np.int64)

    def find_x_center(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        tr_state: Any,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        del x_obs, y_obs, tr_state, rng
        return None

    def predict(self, x: np.ndarray) -> PosteriorResult:
        return self._predict(x, exclude_nearest=False)

    def predict_leave_one_out(self, x: np.ndarray) -> PosteriorResult:
        model = self._require_model()
        if model.train_y.shape[0] <= 1:
            return self.predict(x)
        return self._predict(x, exclude_nearest=True)

    def _predict(self, x: np.ndarray, *, exclude_nearest: bool) -> PosteriorResult:
        x_arr = np.asarray(x, dtype=float)
        model = self._require_model()
        num_neighbors_available = model.train_y.shape[0] - int(bool(exclude_nearest))
        k_eff = _effective_k(self._config.k, num_neighbors_available)
        dist2, idx = model.enn_model.rust_backend.neighbor_distances_and_indices(
            x_arr,
            k_eff,
            exclude_nearest=bool(exclude_nearest),
        )
        weights, base_sigma = _enn_weighted_posterior_scale(
            np.asarray(dist2, dtype=float),
            np.asarray(idx, dtype=np.int64),
            model,
            self._config,
        )
        mu = np.sum(weights * model.train_y[np.asarray(idx, dtype=np.int64)], axis=1)
        varentropy = _weight_varentropy(weights, normalize=bool(self._config.normalize_varentropy))
        sigma = base_sigma * (1.0 + float(self._config.varentropy_scale) * varentropy)
        return PosteriorResult(mu=mu.reshape(-1, 1), sigma=sigma.reshape(-1, 1))

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        post = self.predict(x)
        sigma = np.zeros_like(post.mu) if post.sigma is None else np.asarray(post.sigma, dtype=float)
        draws = rng.normal(
            loc=np.asarray(post.mu, dtype=float)[None, :, :],
            scale=np.maximum(sigma, 0.0)[None, :, :],
            size=(int(num_samples), post.mu.shape[0], post.mu.shape[1]),
        )
        return np.asarray(draws, dtype=float)

    def _require_model(self) -> _FittedENNVarentropyModel:
        if self._model is None:
            raise RuntimeError("ENNVarentropySurrogate requires a fitted model")
        return self._model


def _enn_weighted_posterior_scale(
    dist2: np.ndarray,
    idx: np.ndarray,
    model: _FittedENNVarentropyModel,
    cfg: ENNVarentropySurrogateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    local_var = np.maximum(dist2, float(cfg.variance_eps)) * float(model.y_scale) ** 2
    if bool(cfg.include_noise_in_sigma) and model.train_yvar is not None:
        local_var = local_var + np.maximum(model.train_yvar[idx], float(cfg.variance_eps))
    precision = 1.0 / np.maximum(local_var, float(cfg.variance_eps))
    precision_sum = np.sum(precision, axis=1, keepdims=True)
    weights = precision / np.maximum(precision_sum, float(cfg.variance_eps))
    sigma = np.sqrt(1.0 / np.maximum(precision_sum[:, 0], float(cfg.variance_eps)))
    return weights, sigma


def _can_append(
    model: _FittedENNVarentropyModel | None,
    x: np.ndarray,
    y: np.ndarray,
    yv: np.ndarray | None,
) -> bool:
    if model is None:
        return False
    n_old = model.train_x.shape[0]
    if x.shape[0] < n_old or x.shape[1:] != model.train_x.shape[1:]:
        return False
    if not np.array_equal(model.train_x, x[:n_old]):
        return False
    if not np.array_equal(model.train_y, y[:n_old]):
        return False
    if model.train_yvar is None or yv is None:
        return model.train_yvar is None and yv is None
    return np.array_equal(model.train_yvar, yv[:n_old])


def _append_training_rows(
    model: _FittedENNVarentropyModel,
    x: np.ndarray,
    y: np.ndarray,
    yv: np.ndarray | None,
) -> None:
    n_old = model.train_x.shape[0]
    if x.shape[0] > n_old:
        yv_new = None if yv is None else yv[n_old:].reshape(-1, 1)
        model.enn_model.add(x[n_old:], y[n_old:].reshape(-1, 1), yv_new)
        model.enn_model.ensure_index_sync()
    model.train_x = x.copy()
    model.train_y = y.copy()
    model.train_yvar = None if yv is None else yv.copy()
    model.y_scale = _y_scale(y)


def _weight_varentropy(weights: np.ndarray, *, normalize: bool) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    eps = np.finfo(float).tiny
    surprisal = -np.log(np.maximum(w, eps))
    entropy = np.sum(w * surprisal, axis=1, keepdims=True)
    varentropy = np.sum(w * np.square(surprisal - entropy), axis=1)
    if not normalize:
        return np.maximum(varentropy, 0.0)
    k_eff = max(1, w.shape[1])
    if k_eff <= 1:
        return np.zeros(w.shape[0], dtype=float)
    denom = max(float(np.log(k_eff)) ** 2, eps)
    return np.clip(varentropy / denom, 0.0, 1.0)


def _coerce_scalar_y(y_obs: np.ndarray) -> np.ndarray:
    y = np.asarray(y_obs, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError(f"ENNVarentropySurrogate supports scalar y only, got shape {y.shape}")


def _coerce_yvar(y_var: np.ndarray | None, n: int) -> np.ndarray | None:
    if y_var is None:
        return None
    arr = np.asarray(y_var, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or arr.shape[0] != int(n):
        raise ValueError(f"y_var must have shape ({n},) or ({n}, 1), got {arr.shape}")
    if not np.all(np.isfinite(arr) & (arr >= 0.0)):
        raise ValueError("y_var must be finite and non-negative")
    return arr


def _effective_k(k: int, num_obs: int) -> int:
    return max(1, min(int(k), max(1, int(num_obs))))


def _y_scale(y: np.ndarray) -> float:
    scale = float(np.std(np.asarray(y, dtype=float)))
    return max(scale, 0.0)


__all__ = ["ENNVarentropySurrogate", "ENNVarentropySurrogateConfig", "_weight_varentropy"]
