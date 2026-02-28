"""PufferLib PPO configuration."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PufferPPOConfig:
    exp_dir: str = "_tmp/atari/ppo_puffer"
    env_tag: str = "atari:Pong"
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None

    total_timesteps: int = 1000000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    backbone_name: str = "nature_cnn_atari"
    backbone_hidden_sizes: tuple[int, ...] = ()
    backbone_activation: str = "relu"
    backbone_layer_norm: bool = False
    actor_head_hidden_sizes: tuple[int, ...] = (512,)
    critic_head_hidden_sizes: tuple[int, ...] = (512,)
    head_activation: str = "relu"
    share_backbone: bool = True
    log_std_init: float = -0.5

    device: str = "auto"
    vector_backend: str = "serial"  # serial | multiprocessing
    vector_num_workers: int | None = None
    vector_batch_size: int | None = None
    vector_overwork: bool = False
    framestack: int = 4
    pixels_only: bool = True

    eval_interval: int = 1
    num_denoise_eval: int | None = None
    num_denoise_passive_eval: int | None = None
    eval_seed_base: int | None = None
    eval_noise_mode: str | None = None
    log_interval: int = 1
    checkpoint_interval: int | None = None
    resume_from: str | None = None
    video_enable: bool = False
    video_prefix: str = "policy"
    video_num_episodes: int = 10
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: int | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "PufferPPOConfig":
        data = dict(raw)
        for key in [
            "backbone_hidden_sizes",
            "actor_head_hidden_sizes",
            "critic_head_hidden_sizes",
        ]:
            if key in data and data[key] is not None:
                data[key] = tuple(int(x) for x in data[key])
        return cls(**data)


@dataclasses.dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    last_heldout_return: float | None
    num_iterations: int
