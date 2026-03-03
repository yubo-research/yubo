from __future__ import annotations

import dataclasses
from typing import ClassVar

from rl.core.config_utils import DataclassFromDictMixin


@dataclasses.dataclass
class WPOConfig(DataclassFromDictMixin):
    _tuple_int_keys: ClassVar[tuple[str, ...]] = ("backbone_hidden_sizes", "actor_head_hidden_sizes", "critic_head_hidden_sizes")

    exp_dir: str = "_tmp/wpo_puffer"
    env_tag: str = "pend"
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None
    from_pixels: bool = False
    pixels_only: bool = True
    total_timesteps: int = 1000000
    num_envs: int = 1
    frames_per_batch: int = 4
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    learning_rate_dual: float = 0.0003
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    replay_size: int = 1000000
    replay_backend: str = "auto"
    learning_starts: int = 5000
    update_every: int = 1
    updates_per_step: int = 1
    alpha_init: float = 0.2
    num_samples: int = 20
    epsilon_mean: float = 0.0025
    epsilon_stddev: float = 1e-6
    init_log_alpha_mean: float = 0.0
    init_log_alpha_stddev: float = 2.0
    policy_loss_scale: float = 1.0
    kl_loss_scale: float = 1.0
    dual_loss_scale: float = 1.0
    per_dim_constraining: bool = True
    squashing_type: str = "identity"
    eval_interval_steps: int = 10000
    num_denoise_eval: int | None = None
    num_denoise_passive_eval: int | None = None
    eval_seed_base: int | None = None
    eval_noise_mode: str | None = None
    backbone_name: str = "mlp"
    backbone_hidden_sizes: tuple[int, ...] = (256, 256)
    backbone_activation: str = "silu"
    backbone_layer_norm: bool = False
    actor_head_hidden_sizes: tuple[int, ...] = ()
    critic_head_hidden_sizes: tuple[int, ...] = ()
    head_activation: str = "silu"
    theta_dim: int | None = None
    device: str = "auto"
    collector_backend: str = "auto"
    single_env_backend: str = "auto"
    collector_workers: int | None = None
    vector_backend: str = "serial"
    vector_num_workers: int | None = None
    vector_batch_size: int | None = None
    vector_overwork: bool = False
    framestack: int = 1
    log_interval_steps: int = 1000
    checkpoint_interval_steps: int | None = None
    resume_from: str | None = None
    video_enable: bool = False
    video_prefix: str = "policy"
    video_num_episodes: int = 10
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: int | None = None

    def runtime_num_envs(self) -> int:
        return int(self.num_envs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    last_heldout_return: float | None
    num_steps: int


__all__ = ["WPOConfig", "TrainResult"]
