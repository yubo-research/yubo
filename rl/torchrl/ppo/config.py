from __future__ import annotations

import dataclasses

from rl.config_model_defaults import apply_ppo_env_model_defaults
from rl.core.torchrl_runtime import TorchRLRuntimeCapabilities, TorchRLRuntimeConfig


@dataclasses.dataclass
class PPOConfig(TorchRLRuntimeConfig):
    exp_dir: str = "_tmp/ppo"
    env_tag: str = "pend"
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None
    from_pixels: bool = False
    pixels_only: bool = True
    total_timesteps: int = 1000000
    learning_rate: float = 0.0003
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    eval_interval: int = 1
    num_denoise: int | None = None
    num_denoise_passive: int | None = None
    eval_seed_base: int | None = None
    eval_noise_mode: str | None = None
    backbone_name: str = "mlp"
    backbone_hidden_sizes: tuple[int, ...] = (64, 64)
    backbone_activation: str = "silu"
    backbone_layer_norm: bool = True
    actor_head_hidden_sizes: tuple[int, ...] = ()
    critic_head_hidden_sizes: tuple[int, ...] = ()
    head_activation: str = "silu"
    share_backbone: bool = True
    log_std_init: float = 0.0
    theta_dim: int | None = None
    log_interval: int = 1
    checkpoint_interval: int | None = None
    resume_from: str | None = None
    video_enable: bool = False
    video_prefix: str = "policy"
    video_num_episodes: int = 10
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: int | None = None
    profile_enable: bool = False
    profile_wait: int = 0
    profile_warmup: int = 1
    profile_active: int = 3

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "PPOConfig":
        d = apply_ppo_env_model_defaults(raw)
        return cls(**d)

    def runtime_num_envs(self) -> int:
        return int(self.num_envs)


_PPO_RUNTIME_CAPABILITIES = TorchRLRuntimeCapabilities(
    allow_multi_sync_collector=True, allow_multi_async_collector=True, allow_mps_multi_collectors=False, allow_parallel_single_env=True
)


@dataclasses.dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    last_heldout_return: float | None
    num_iterations: int
