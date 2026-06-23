from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class MarsSurrogateConfig:
    max_terms: int = 64
    interaction_order: int = 2
    num_bootstrap: int = 8
    active_rank: int = 8
    trailing_obs: int | None = 256
    feature_screen: int = 512
    knots_per_feature: int = 3
    ridge: float = 1e-6
    active_samples: int = 256
    lam_min: float = 1e-4
    lam_max: float = 1e4
    eps: float = 1e-6
    kappa_max: float = 1e4

    def __post_init__(self) -> None:
        _check_at_least("max_terms", self.max_terms, 1)
        if self.interaction_order not in (1, 2):
            raise ValueError("interaction_order must be 1 or 2")
        _check_at_least("num_bootstrap", self.num_bootstrap, 1)
        _check_at_least("active_rank", self.active_rank, 1)
        if self.trailing_obs is not None:
            _check_at_least("trailing_obs", self.trailing_obs, 1)
        _check_at_least("feature_screen", self.feature_screen, 1)
        _check_at_least("knots_per_feature", self.knots_per_feature, 1)
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        _check_at_least("active_samples", self.active_samples, 1)


@dataclass(frozen=True)
class BayesianMarsSurrogateConfig:
    basis: MarsSurrogateConfig = field(default_factory=MarsSurrogateConfig)
    prior_precision: float = 1.0
    intercept_prior_precision: float = 1e-8
    noise_variance: float | None = None
    min_noise_variance: float = 1e-8
    include_noise_in_sigma: bool = False
    basis_refresh_interval: int = 1
    posterior_jitter: float = 1e-10
    basis_sampler: str = "deterministic"
    mcmc_steps: int = 128
    mcmc_burn_in: int = 32
    mcmc_thin: int = 4
    mcmc_num_models: int = 16
    mcmc_pool_size: int | None = None
    mcmc_term_prior: float | None = None

    def __post_init__(self) -> None:
        if self.prior_precision <= 0.0:
            raise ValueError("prior_precision must be > 0")
        if self.intercept_prior_precision < 0.0:
            raise ValueError("intercept_prior_precision must be >= 0")
        if self.noise_variance is not None and self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be > 0 when provided")
        if self.min_noise_variance <= 0.0:
            raise ValueError("min_noise_variance must be > 0")
        _check_at_least("basis_refresh_interval", self.basis_refresh_interval, 1)
        if self.posterior_jitter < 0.0:
            raise ValueError("posterior_jitter must be >= 0")
        if self.basis_sampler not in {"deterministic", "mcmc"}:
            raise ValueError("basis_sampler must be 'deterministic' or 'mcmc'")
        _check_at_least("mcmc_steps", self.mcmc_steps, 1)
        _check_at_least("mcmc_burn_in", self.mcmc_burn_in, 0)
        _check_at_least("mcmc_thin", self.mcmc_thin, 1)
        _check_at_least("mcmc_num_models", self.mcmc_num_models, 1)
        if self.mcmc_pool_size is not None:
            _check_at_least("mcmc_pool_size", self.mcmc_pool_size, 1)
        if self.mcmc_term_prior is not None and not (0.0 < self.mcmc_term_prior < 1.0):
            raise ValueError("mcmc_term_prior must be in (0, 1) when provided")

    @property
    def trailing_obs(self) -> int | None:
        return self.basis.trailing_obs

    @property
    def active_rank(self) -> int:
        return self.basis.active_rank

    @property
    def active_samples(self) -> int:
        return self.basis.active_samples

    def with_active_rank(self, active_rank: int) -> BayesianMarsSurrogateConfig:
        return replace(self, basis=replace(self.basis, active_rank=int(active_rank)))


@dataclass(frozen=True)
class ENNMarsGeometrySurrogateConfig:
    enn: Any
    mars: MarsSurrogateConfig


def _check_at_least(name: str, value: int, minimum: int) -> None:
    if int(value) < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
