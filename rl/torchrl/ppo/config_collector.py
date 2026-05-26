from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PPOCollectorConfig:
    total_frames: int = 1000000
    frames_per_batch: int = 2048
    num_envs: int = 1
    backend: str = "auto"
    single_env_backend: str = "auto"
    workers: int | None = None


@dataclasses.dataclass
class PPOOptimConfig:
    lr: float = 0.0003
    anneal_lr: bool = True
    minibatch_size: int = 64
    num_epochs: int = 10
    max_grad_norm: float = 0.5


@dataclasses.dataclass
class PPOLossConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    clip_epsilon: float = 0.2
    clip_value_loss: bool = True
    entropy_coeff: float = 0.0
    critic_coeff: float = 0.5
    target_kl: float | None = None
