from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SACCollectorConfig:
    total_frames: int = 1000000
    frames_per_batch: int = 4
    init_random_frames: int = 5000
    num_envs: int = 1
    backend: str = "auto"
    single_env_backend: str = "auto"
    workers: int | None = None


@dataclasses.dataclass
class SACReplayBufferConfig:
    size: int = 1000000
    batch_size: int = 256
    pin_memory: bool = False
    prefetch: int | None = None


@dataclasses.dataclass
class SACOptimConfig:
    actor_lr: float = 0.0003
    qvalue_lr: float = 0.0003
    alpha_lr: float = 0.0003
    update_every: int = 1
    optim_steps_per_batch: int = 1
    learner_update_chunk_size: int = 1
