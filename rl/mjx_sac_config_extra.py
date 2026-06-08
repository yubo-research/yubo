from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MJXSACEvalConfig:
    interval: int = 10
    num_envs: int = 128
    num_steps: int = 1000


@dataclasses.dataclass
class MJXSACCheckpointConfig:
    interval: int | None = 100
    resume_from: str | None = None
