from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MJXPPOEvalConfig:
    interval: int = 10
    num_envs: int = 128
    num_steps: int = 1000


@dataclasses.dataclass
class MJXPPOCheckpointConfig:
    interval: int | None = 100
    resume_from: str | None = None
