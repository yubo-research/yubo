from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SACCheckpointConfig:
    interval_steps: int | None = None
    resume_from: str | None = None
