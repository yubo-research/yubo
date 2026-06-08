from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PPOEvalConfig:
    interval: int = 1
    num_denoise: int | None = None
    num_denoise_passive: int | None = None
    seed_base: int | None = None
    noise_mode: str | None = None


@dataclasses.dataclass
class PPOCheckpointConfig:
    interval: int | None = None
    resume_from: str | None = None


@dataclasses.dataclass
class PPOProfileConfig:
    enable: bool = False
    wait: int = 0
    warmup: int = 1
    active: int = 3
