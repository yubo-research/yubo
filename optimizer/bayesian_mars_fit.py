from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars_basis import _MarsTerm, _ridge_solve, _safe_solve, _standardize_y, _standardized_y_var
from .mars_config import BayesianMarsSurrogateConfig
from .mars_fit import _fit_single_mars, _FittedMarsModel
from .mars_geometry import _LowRankFactor


@dataclass
class _FittedBayesianMarsModel:
    basis_model: _FittedMarsModel
    coef_mean: np.ndarray
    coef_cov: np.ndarray
    noise_variance: float
    include_noise_in_sigma: bool
    posterior_jitter: float

    def _design(self, x: np.ndarray) -> np.ndarray:
        return self.basis_model._design(x)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = self._design(x)
        mu_std = phi @ self.coef_mean
        latent_var = np.einsum("ij,jk,ik->i", phi, self.coef_cov, phi)
        latent_var = np.maximum(latent_var, 0.0)
        if self.include_noise_in_sigma:
            latent_var = latent_var + float(self.noise_variance)
        mu = float(self.basis_model.y_mean) + float(self.basis_model.y_scale) * mu_std
        sigma = float(self.basis_model.y_scale) * np.sqrt(latent_var)
        return mu, sigma

    def sample(
        self,
        x: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        phi = self._design(x)
        cov = _stable_cov(self.coef_cov, self.posterior_jitter)
        draws = rng.multivariate_normal(
            self.coef_mean,
            cov,
            size=int(num_samples),
            check_valid="ignore",
        )
        samples_std = draws @ phi.T
        return float(self.basis_model.y_mean) + float(self.basis_model.y_scale) * samples_std

    def active_low_rank_factor(
        self,
        cfg: BayesianMarsSurrogateConfig,
        rng: np.random.Generator,
    ) -> _LowRankFactor | None:
        mean_model = _FittedMarsModel(
            terms=self.basis_model.terms,
            coef=self.coef_mean,
            y_mean=self.basis_model.y_mean,
            y_scale=self.basis_model.y_scale,
            num_dim=self.basis_model.num_dim,
        )
        return mean_model.active_low_rank_factor(cfg.basis, rng)


def _stable_cov(coef_cov: np.ndarray, posterior_jitter: float) -> np.ndarray:
    cov = 0.5 * (coef_cov + coef_cov.T)
    if posterior_jitter > 0.0:
        cov = cov + float(posterior_jitter) * np.eye(cov.shape[0], dtype=float)
    return cov


def _fit_bayesian_mars(
    x: np.ndarray,
    y: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    *,
    terms: tuple[_MarsTerm, ...] | None = None,
    y_var: np.ndarray | None = None,
) -> _FittedBayesianMarsModel:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_std, y_mean, y_scale = _standardize_y(y_arr)
    basis_model = _basis_model(x_arr, y_arr, y_mean, y_scale, cfg, terms)
    phi = basis_model._design(x_arr)
    noise_variance = _noise_variance(phi, y_std, y_var, y_scale, cfg)
    precision, rhs = _posterior_system(phi, y_std, cfg, noise_variance)
    coef_mean = _safe_solve(precision, rhs)
    coef_cov = _safe_solve(precision, np.eye(precision.shape[0], dtype=float))
    return _FittedBayesianMarsModel(
        basis_model=basis_model,
        coef_mean=np.asarray(coef_mean, dtype=float).reshape(-1),
        coef_cov=0.5 * (np.asarray(coef_cov, dtype=float) + np.asarray(coef_cov, dtype=float).T),
        noise_variance=noise_variance,
        include_noise_in_sigma=bool(cfg.include_noise_in_sigma),
        posterior_jitter=float(cfg.posterior_jitter),
    )


def _basis_model(
    x: np.ndarray,
    y: np.ndarray,
    y_mean: float,
    y_scale: float,
    cfg: BayesianMarsSurrogateConfig,
    terms: tuple[_MarsTerm, ...] | None,
) -> _FittedMarsModel:
    if terms is None:
        fitted = _fit_single_mars(x, y, cfg.basis)
        terms = fitted.terms
    return _FittedMarsModel(
        terms=tuple(terms),
        coef=np.zeros((len(terms) + 1,), dtype=float),
        y_mean=y_mean,
        y_scale=y_scale,
        num_dim=int(x.shape[1]),
    )


def _noise_variance(
    phi: np.ndarray,
    y_std: np.ndarray,
    y_var: np.ndarray | None,
    y_scale: float,
    cfg: BayesianMarsSurrogateConfig,
) -> float:
    y_var_std = _standardized_y_var(y_var, y_scale)
    if y_var_std is not None:
        noise_variance = float(np.mean(y_var_std))
    elif cfg.noise_variance is None:
        noise_variance = _residual_noise_variance(phi, y_std, cfg)
    else:
        noise_variance = float(cfg.noise_variance)
    if not np.isfinite(noise_variance):
        noise_variance = float(cfg.min_noise_variance)
    return max(noise_variance, float(cfg.min_noise_variance))


def _residual_noise_variance(
    phi: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
) -> float:
    coef_ridge = _ridge_solve(phi, y_std, float(cfg.basis.ridge))
    residual = y_std - phi @ coef_ridge
    dof = max(int(phi.shape[0]) - int(phi.shape[1]), 1)
    return float(np.sum(residual**2) / float(dof))


def _posterior_system(
    phi: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    prior_precision = float(cfg.prior_precision) * np.ones((phi.shape[1],), dtype=float)
    prior_precision[0] = float(cfg.intercept_prior_precision)
    precision = np.diag(prior_precision) + (phi.T @ phi) / noise_variance
    if cfg.posterior_jitter > 0.0:
        precision = precision + float(cfg.posterior_jitter) * np.eye(precision.shape[0], dtype=float)
    rhs = (phi.T @ y_std) / noise_variance
    return precision, rhs
