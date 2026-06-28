from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.python_fallback.components import PosteriorResult, SurrogateResult

from .enn_index_driver import parse_enn_index_driver
from .mars_basis import _coerce_scalar_y, _standardize_y
from .mars_enn_config import MarsENNSurrogateConfig
from .mars_fit import _fit_single_mars, _FittedMarsModel
from .mars_geometry import _LowRankFactor
from .mars_surrogate import _set_low_rank_factor, _trim_observations
from .uhd_enn_fit_helpers import fit_enn_params


@dataclass
class _FittedMarsENNModel:
    basis_model: _FittedMarsModel
    enn_model: EpistemicNearestNeighbors
    enn_params: Any
    y_mean: float
    y_scale: float


class MarsENNSurrogate:
    def __init__(self, config: MarsENNSurrogateConfig) -> None:
        self._config = config
        self._model: _FittedMarsENNModel | None = None

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
        y_std, y_mean, y_scale = _standardize_y(y)
        basis_model = _fit_single_mars(x, y, self._config.basis)
        phi = _mars_enn_features(basis_model, x)
        yvar_train = _mars_enn_yvar(yv, y_scale)
        enn_model = EpistemicNearestNeighbors(
            phi,
            y_std.reshape(-1, 1),
            yvar_train,
            scale_x=bool(self._config.scale_x),
            index_driver=parse_enn_index_driver(self._config.index_driver),
        )
        enn_params = fit_enn_params(
            enn_model,
            phi,
            y_std,
            k=_effective_enn_k(self._config.k, x.shape[0]),
            num_fit_candidates=int(self._config.num_fit_candidates),
            num_fit_samples=int(self._config.num_fit_samples),
            rng=rng,
            yvar=yvar_train,
            infer_aleatoric_variance_scale=bool(self._config.infer_aleatoric_variance_scale),
        )
        self._model = _FittedMarsENNModel(
            basis_model=basis_model,
            enn_model=enn_model,
            enn_params=enn_params,
            y_mean=float(y_mean),
            y_scale=float(y_scale),
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
        x_arr = np.asarray(x, dtype=float)
        model = self._require_model()
        phi = _mars_enn_features(model.basis_model, x_arr)
        post = model.enn_model.posterior(
            phi,
            params=model.enn_params,
            flags=PosteriorFlags(observation_noise=bool(self._config.include_noise_in_sigma)),
        )
        mu_std = np.asarray(post.mu, dtype=float).reshape(-1)
        se_std = np.asarray(post.se, dtype=float).reshape(-1)
        mu = float(model.y_mean) + float(model.y_scale) * mu_std
        sigma = abs(float(model.y_scale)) * np.maximum(se_std, 0.0)
        return PosteriorResult(mu=mu.reshape(-1, 1), sigma=sigma.reshape(-1, 1))

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        model = self._require_model()
        phi = _mars_enn_features(model.basis_model, x_arr)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=int(num_samples)).tolist()
        draws_std, _neighbors = model.enn_model.posterior_function_draw(
            phi,
            params=model.enn_params,
            function_seeds=seeds,
            flags=PosteriorFlags(observation_noise=bool(self._config.include_noise_in_sigma)),
        )
        draws = float(model.y_mean) + float(model.y_scale) * np.asarray(draws_std, dtype=float)
        return draws.reshape(int(num_samples), x_arr.shape[0], 1)

    def active_low_rank_factor(self, rng: np.random.Generator | None = None) -> _LowRankFactor | None:
        rng = np.random.default_rng() if rng is None else rng
        return self._require_model().basis_model.active_low_rank_factor(self._config.basis, rng)

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

    def _require_model(self) -> _FittedMarsENNModel:
        if self._model is None:
            raise RuntimeError("MarsENNSurrogate requires a fitted model")
        return self._model


def _mars_enn_features(model: _FittedMarsModel, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    design = model._design(x_arr)
    if design.shape[1] <= 1:
        return x_arr
    return np.asarray(design[:, 1:], dtype=float)


def _mars_enn_yvar(y_var: np.ndarray | None, y_scale: float) -> np.ndarray | None:
    if y_var is None:
        return None
    arr = np.asarray(y_var, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr) & (arr > 0.0)):
        return None
    return (arr / max(float(y_scale) ** 2, 1e-12)).reshape(-1, 1)


def _effective_enn_k(k: int, num_obs: int) -> int:
    return max(1, min(int(k), max(1, int(num_obs))))


__all__ = ["MarsENNSurrogate", "MarsENNSurrogateConfig"]
