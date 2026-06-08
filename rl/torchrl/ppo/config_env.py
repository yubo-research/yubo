from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PPOEnvConfig:
    normalize_observation: bool = False
    normalize_reward: bool = False
    reward_normalize_gamma: float = 0.99
    observation_clip: tuple[float, float] | None = None
    reward_clip: tuple[float, float] | None = None
