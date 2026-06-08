from __future__ import annotations

import dataclasses

from rl.mjx_sac_config_extra import MJXSACCheckpointConfig, MJXSACEvalConfig

__all__ = [
    "MJXSACCheckpointConfig",
    "MJXSACCollectorConfig",
    "MJXSACEvalConfig",
    "MJXSACLossConfig",
    "MJXSACOptimConfig",
]


@dataclasses.dataclass
class MJXSACCollectorConfig:
    total_frames: int = 262_144
    num_envs: int = 1024
    num_steps: int = 1
    replay_size: int = 1_000_000
    batch_size: int = 4096
    updates_per_iter: int = 1


@dataclasses.dataclass
class MJXSACOptimConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4


@dataclasses.dataclass
class MJXSACLossConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha_init: float = 0.2
    target_entropy: float | None = None
