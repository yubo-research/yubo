from __future__ import annotations

import dataclasses

from rl.mjx_ppo_config_extra import MJXPPOCheckpointConfig, MJXPPOEvalConfig

__all__ = [
    "MJXPPOCheckpointConfig",
    "MJXPPOCollectorConfig",
    "MJXPPOEvalConfig",
    "MJXPPOLossConfig",
    "MJXPPOOptimConfig",
]


@dataclasses.dataclass
class MJXPPOCollectorConfig:
    total_frames: int = 1_048_576
    num_envs: int = 2048
    num_steps: int = 32


@dataclasses.dataclass
class MJXPPOOptimConfig:
    lr: float = 3e-4
    anneal_lr: bool = True
    minibatch_size: int = 8192
    num_epochs: int = 4
    max_grad_norm: float = 0.5


@dataclasses.dataclass
class MJXPPOLossConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_reward: bool = True
    clip_epsilon: float = 0.2
    clip_value_loss: bool = True
    entropy_coeff: float = 0.0
    critic_coeff: float = 0.5
