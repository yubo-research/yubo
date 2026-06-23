from __future__ import annotations

from typing import Any

import numpy as np

from .mars_config import BayesianMarsSurrogateConfig, MarsSurrogateConfig
from .mars_surrogate import BayesianMarsSurrogate, MarsSurrogate


def create_mars_optimizer(
    *,
    bounds: np.ndarray,
    config: Any,
    rng: np.random.Generator,
) -> Any:
    from enn.turbo.python_fallback.components.builder import build_acquisition_optimizer
    from enn.turbo.python_fallback.optimizer import Optimizer

    return Optimizer(
        bounds=bounds,
        config=config,
        rng=rng,
        surrogate=_build_mars_surrogate(config.surrogate),
        acquisition_optimizer=build_acquisition_optimizer(config.acquisition),
    )


def _build_mars_surrogate(config: Any) -> MarsSurrogate | BayesianMarsSurrogate:
    if isinstance(config, MarsSurrogateConfig):
        return MarsSurrogate(config)
    if isinstance(config, BayesianMarsSurrogateConfig):
        return BayesianMarsSurrogate(config)
    raise ValueError(f"Unsupported MARS surrogate config: {type(config).__name__}")
