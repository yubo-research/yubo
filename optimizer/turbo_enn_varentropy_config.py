from __future__ import annotations

from dataclasses import dataclass, field

from .enn_varentropy_config import ENNVarentropySurrogateConfig


@dataclass(frozen=True)
class TurboENNVarentropyDesignerConfig:
    enn: ENNVarentropySurrogateConfig = field(default_factory=ENNVarentropySurrogateConfig)
    acq_type: str = "ucb"
    num_init: int | None = None
    num_keep: int | None = None
    num_candidates: int | None = None
    candidate_rv: str | None = None
    tr_type: str | None = None


__all__ = ["TurboENNVarentropyDesignerConfig"]
