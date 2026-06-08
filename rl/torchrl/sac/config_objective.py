from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SACLossConfig:
    gamma: float = 0.99
    alpha_init: float = 0.2
    target_entropy: float | None = None


@dataclasses.dataclass
class SACTargetNetUpdaterConfig:
    tau: float = 0.005


@dataclasses.dataclass
class SACEvalConfig:
    interval_steps: int = 10000
    num_denoise: int | None = None
    num_denoise_passive: int | None = None
    seed_base: int | None = None
    noise_mode: str | None = None
