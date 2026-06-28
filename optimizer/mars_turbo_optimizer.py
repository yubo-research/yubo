from __future__ import annotations

from typing import Any

import numpy as np

from .enn_varentropy_config import ENNVarentropySurrogateConfig
from .enn_varentropy_surrogate import ENNVarentropySurrogate
from .mars_config import BayesianMarsSurrogateConfig, MarsSurrogateConfig
from .mars_enn_config import MarsENNSurrogateConfig
from .mars_enn_surrogate import MarsENNSurrogate
from .mars_surrogate import BayesianMarsSurrogate, MarsSurrogate


def create_mars_optimizer(
    *,
    bounds: np.ndarray,
    config: Any,
    rng: np.random.Generator,
) -> Any:
    return create_custom_surrogate_optimizer(bounds=bounds, config=config, rng=rng)


def create_custom_surrogate_optimizer(
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
        surrogate=_build_custom_surrogate(config.surrogate),
        acquisition_optimizer=build_acquisition_optimizer(config.acquisition),
    )


def _build_custom_surrogate(
    config: Any,
) -> MarsSurrogate | BayesianMarsSurrogate | MarsENNSurrogate | ENNVarentropySurrogate:
    if isinstance(config, MarsSurrogateConfig):
        return MarsSurrogate(config)
    if isinstance(config, BayesianMarsSurrogateConfig):
        return BayesianMarsSurrogate(config)
    if isinstance(config, MarsENNSurrogateConfig):
        return MarsENNSurrogate(config)
    if isinstance(config, ENNVarentropySurrogateConfig):
        return ENNVarentropySurrogate(config)
    raise ValueError(f"Unsupported custom surrogate config: {type(config).__name__}")


_build_mars_surrogate = _build_custom_surrogate
