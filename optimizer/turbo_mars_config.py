from __future__ import annotations

from dataclasses import dataclass, field

from .mars_config import BayesianMarsSurrogateConfig, MarsSurrogateConfig
from .mars_enn_config import MarsENNSurrogateConfig


@dataclass(frozen=True)
class TurboMARSDesignerConfig:
    mars: MarsSurrogateConfig = field(default_factory=MarsSurrogateConfig)
    acq_type: str = "ucb"
    num_init: int | None = None
    num_keep: int | None = None
    num_candidates: int | None = None
    candidate_rv: str | None = None
    tr_type: str | None = None


@dataclass(frozen=True)
class TurboBayesianMARSDesignerConfig:
    bmars: BayesianMarsSurrogateConfig = field(default_factory=BayesianMarsSurrogateConfig)
    acq_type: str = "ucb"
    num_init: int | None = None
    num_keep: int | None = None
    num_candidates: int | None = None
    candidate_rv: str | None = None
    tr_type: str | None = None


@dataclass(frozen=True)
class TurboMarsENNDesignerConfig:
    mars_enn: MarsENNSurrogateConfig = field(default_factory=MarsENNSurrogateConfig)
    acq_type: str = "ucb"
    num_init: int | None = None
    num_keep: int | None = None
    num_candidates: int | None = None
    candidate_rv: str | None = None
    tr_type: str | None = None


def stable_bmars_config() -> BayesianMarsSurrogateConfig:
    return BayesianMarsSurrogateConfig(
        basis=MarsSurrogateConfig(
            max_terms=16,
            interaction_order=1,
            num_bootstrap=1,
            active_rank=4,
            trailing_obs=32,
            feature_screen=None,
            knots_per_feature=None,
            active_samples=32,
        ),
        include_noise_in_sigma=True,
        basis_sampler="mcmc",
        mcmc_steps=32,
        mcmc_burn_in=32,
        mcmc_thin=4,
        mcmc_num_models=16,
        mcmc_pool_size=32,
        mcmc_term_prior=0.125,
    )
