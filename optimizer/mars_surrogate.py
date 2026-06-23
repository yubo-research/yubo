from __future__ import annotations

from typing import Any

import numpy as np
from enn.turbo.python_fallback.components import PosteriorResult, SurrogateResult

from .bayesian_mars_fit import _FittedBayesianMarsModel, _fit_bayesian_mars
from .bayesian_mars_mcmc import _fit_bayesian_mars_mcmc
from .mars_basis import _HingeFactor, _MarsTerm, _build_main_basis, _coerce_scalar_y
from .mars_config import BayesianMarsSurrogateConfig, ENNMarsGeometrySurrogateConfig, MarsSurrogateConfig
from .mars_fit import _FittedMarsModel, _fit_single_mars
from .mars_geometry import _LowRankFactor


class MarsSurrogate:
    def __init__(self, config: MarsSurrogateConfig) -> None:
        self._config = config
        self._model: _FittedMarsModel | None = None
        self._ensemble: list[_FittedMarsModel] = []

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
        return self._fit(x_obs, y_obs, y_var, num_steps=num_steps, rng=rng, fit_bootstrap=True)

    def fit_geometry(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: np.random.Generator | None = None,
    ) -> SurrogateResult:
        return self._fit(x_obs, y_obs, y_var, num_steps=num_steps, rng=rng, fit_bootstrap=False)

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
        x_arr = np.asarray(x, dtype=float)
        self._require_model()
        preds = np.column_stack([model.predict(x_arr) for model in self._ensemble])
        mu = np.mean(preds, axis=1).reshape(-1, 1)
        sigma = _ensemble_sigma(preds).reshape(-1, 1)
        return PosteriorResult(mu=mu, sigma=sigma)

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        self._require_model()
        choices = rng.integers(0, len(self._ensemble), size=int(num_samples))
        out = np.empty((int(num_samples), x_arr.shape[0], 1), dtype=float)
        for i, model_idx in enumerate(choices):
            out[i, :, 0] = self._ensemble[int(model_idx)].predict(x_arr)
        return out

    def active_low_rank_factor(self, rng: np.random.Generator | None = None) -> _LowRankFactor | None:
        rng = np.random.default_rng() if rng is None else rng
        return self._require_model().active_low_rank_factor(self._config, rng)

    def update_trust_region(
        self,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int | None,
        rng: np.random.Generator,
    ) -> None:
        del x_center, y_obs, incumbent_idx
        _set_low_rank_factor(tr_state, self.active_low_rank_factor(rng))

    def _fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: np.random.Generator | None = None,
        fit_bootstrap: bool = True,
    ) -> SurrogateResult:
        del y_var, num_steps
        rng = np.random.default_rng() if rng is None else rng
        x, y, _ = _trim_observations(x_obs, y_obs, None, self._config.trailing_obs)
        self._model = _fit_single_mars(x, y, self._config)
        self._ensemble = [self._model]
        if fit_bootstrap and x.shape[0] >= 3:
            self._ensemble.extend(_bootstrap_models(x, y, self._config, rng))
        return SurrogateResult(model=self._model, lengthscales=None)

    def _require_model(self) -> _FittedMarsModel:
        if self._model is None:
            raise RuntimeError("MarsSurrogate requires a fitted model")
        return self._model


class BayesianMarsSurrogate:
    def __init__(self, config: BayesianMarsSurrogateConfig) -> None:
        self._config = config
        self._model: _FittedBayesianMarsModel | None = None
        self._models: tuple[_FittedBayesianMarsModel, ...] = ()
        self._model_weights: np.ndarray = np.ones((0,), dtype=float)
        self._terms: tuple[_MarsTerm, ...] | None = None
        self._term_sets: tuple[tuple[_MarsTerm, ...], ...] | None = None
        self._fit_count = 0

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
        del num_steps
        rng = np.random.default_rng() if rng is None else rng
        x, y, yv = _trim_observations(x_obs, y_obs, y_var, self._config.trailing_obs)
        if self._config.basis_sampler == "mcmc":
            return self._fit_mcmc(x, y, yv, rng)
        return self._fit_deterministic(x, y, yv)

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
        x_arr = np.asarray(x, dtype=float)
        models, weights = self._require_models()
        if len(models) == 1:
            mu, sigma = models[0].predict(x_arr)
            return PosteriorResult(mu=mu.reshape(-1, 1), sigma=sigma.reshape(-1, 1))
        return _model_averaged_posterior(models, weights, x_arr)

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        models, weights = self._require_models()
        choices = rng.choice(len(models), size=int(num_samples), p=weights)
        out = np.empty((int(num_samples), x_arr.shape[0], 1), dtype=float)
        for i, model_idx in enumerate(choices):
            out[i, :, 0] = models[int(model_idx)].sample(x_arr, 1, rng).reshape(x_arr.shape[0])
        return out

    def active_low_rank_factor(self, rng: np.random.Generator | None = None) -> _LowRankFactor | None:
        rng = np.random.default_rng() if rng is None else rng
        return self._require_model().active_low_rank_factor(self._config, rng)

    def update_trust_region(
        self,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int | None,
        rng: np.random.Generator,
    ) -> None:
        del x_center, y_obs, incumbent_idx
        _set_low_rank_factor(tr_state, self.active_low_rank_factor(rng))

    def _fit_mcmc(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_var: np.ndarray | None,
        rng: np.random.Generator,
    ) -> SurrogateResult:
        result = _fit_bayesian_mars_mcmc(x, y, self._config, rng=rng, y_var=y_var)
        self._models = result.models
        self._model_weights = np.asarray(result.weights, dtype=float).reshape(-1)
        self._model = self._models[int(np.argmax(self._model_weights))]
        self._terms = self._model.basis_model.terms
        self._term_sets = tuple(model.basis_model.terms for model in self._models)
        self._fit_count += 1
        return SurrogateResult(model=self._model, lengthscales=None)

    def _fit_deterministic(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_var: np.ndarray | None,
    ) -> SurrogateResult:
        self._model = _fit_bayesian_mars(x, y, self._config, terms=self._reused_terms(x), y_var=y_var)
        self._models = (self._model,)
        self._model_weights = np.ones((1,), dtype=float)
        self._terms = self._model.basis_model.terms
        self._term_sets = (self._terms,)
        self._fit_count += 1
        return SurrogateResult(model=self._model, lengthscales=None)

    def _reused_terms(self, x: np.ndarray) -> tuple[_MarsTerm, ...] | None:
        refresh = (
            self._terms is None
            or self._model is None
            or self._model.basis_model.num_dim != int(x.shape[1])
            or self._fit_count % int(self._config.basis_refresh_interval) == 0
        )
        return None if refresh else self._terms

    def _require_model(self) -> _FittedBayesianMarsModel:
        if self._model is None:
            raise RuntimeError("BayesianMarsSurrogate requires a fitted model")
        return self._model

    def _require_models(self) -> tuple[tuple[_FittedBayesianMarsModel, ...], np.ndarray]:
        if not self._models:
            raise RuntimeError("BayesianMarsSurrogate requires a fitted model")
        return self._models, _normalized_weights(self._model_weights, len(self._models))


class ENNMarsGeometrySurrogate:
    def __init__(self, config: ENNMarsGeometrySurrogateConfig) -> None:
        from optimizer.enn_surrogate_ext import GeometryENNSurrogate

        self._enn = GeometryENNSurrogate(config.enn)
        self._mars = MarsSurrogate(config.mars)

    @property
    def lengthscales(self) -> np.ndarray | None:
        return self._enn.lengthscales

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: np.random.Generator | None = None,
    ) -> SurrogateResult:
        result = self._enn.fit(x_obs, y_obs, y_var, num_steps=num_steps, rng=rng)
        self._mars.fit_geometry(x_obs, y_obs, y_var, num_steps=num_steps, rng=rng)
        return result

    def get_incumbent_candidate_indices(self, y_obs: np.ndarray) -> np.ndarray:
        return self._enn.get_incumbent_candidate_indices(y_obs)

    def find_x_center(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        tr_state: Any,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        return self._enn.find_x_center(x_obs, y_obs, tr_state, rng)

    def predict(self, x: np.ndarray) -> PosteriorResult:
        return self._enn.predict(x)

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return self._enn.sample(x, num_samples, rng)

    def update_trust_region(
        self,
        tr_state: Any,
        x_center: np.ndarray,
        y_obs: np.ndarray,
        incumbent_idx: int | None,
        rng: np.random.Generator,
    ) -> None:
        self._mars.update_trust_region(tr_state, x_center, y_obs, incumbent_idx, rng)


def _trim_observations(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    y_var: np.ndarray | None,
    trailing_obs: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = np.asarray(x_obs, dtype=float)
    y = _coerce_scalar_y(y_obs)
    yv = None if y_var is None else np.asarray(y_var, dtype=float).reshape(-1)
    if trailing_obs is None or x.shape[0] <= int(trailing_obs):
        return x, y, yv
    keep = int(trailing_obs)
    return x[-keep:], y[-keep:], None if yv is None else yv[-keep:]


def _bootstrap_models(
    x: np.ndarray,
    y: np.ndarray,
    cfg: MarsSurrogateConfig,
    rng: np.random.Generator,
) -> list[_FittedMarsModel]:
    models: list[_FittedMarsModel] = []
    for _ in range(max(0, int(cfg.num_bootstrap) - 1)):
        idx = rng.integers(0, x.shape[0], size=x.shape[0])
        models.append(_fit_single_mars(x[idx], y[idx], cfg))
    return models


def _ensemble_sigma(preds: np.ndarray) -> np.ndarray:
    if preds.shape[1] <= 1:
        return np.zeros((preds.shape[0],), dtype=float)
    return np.std(preds, axis=1, ddof=1)


def _normalized_weights(weights: np.ndarray, num_models: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.shape[0] != int(num_models) or not np.isfinite(arr).all() or float(np.sum(arr)) <= 0.0:
        return np.ones((int(num_models),), dtype=float) / float(num_models)
    return arr / float(np.sum(arr))


def _model_averaged_posterior(
    models: tuple[_FittedBayesianMarsModel, ...],
    weights: np.ndarray,
    x: np.ndarray,
) -> PosteriorResult:
    mus = []
    variances = []
    for model in models:
        mu_i, sigma_i = model.predict(x)
        mus.append(mu_i.reshape(-1))
        variances.append(np.square(sigma_i.reshape(-1)))
    mu_mat = np.vstack(mus)
    var_mat = np.vstack(variances)
    mu = weights @ mu_mat
    second = weights @ (var_mat + np.square(mu_mat))
    sigma = np.sqrt(np.maximum(second - np.square(mu), 0.0))
    return PosteriorResult(mu=mu.reshape(-1, 1), sigma=sigma.reshape(-1, 1))


def _set_low_rank_factor(tr_state: Any, low_rank: _LowRankFactor | None) -> None:
    setter = getattr(tr_state, "set_low_rank_factor", None)
    if callable(setter) and low_rank is not None:
        setter(low_rank)


__all__ = [
    "BayesianMarsSurrogate",
    "BayesianMarsSurrogateConfig",
    "ENNMarsGeometrySurrogate",
    "ENNMarsGeometrySurrogateConfig",
    "MarsSurrogate",
    "MarsSurrogateConfig",
    "_HingeFactor",
    "_MarsTerm",
    "_build_main_basis",
]
