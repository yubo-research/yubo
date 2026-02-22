"""SAC configuration and runtime capabilities."""

from __future__ import annotations

import dataclasses

from ..common.runtime import TorchRLRuntimeCapabilities, TorchRLRuntimeConfig


@dataclasses.dataclass
class SACConfig(TorchRLRuntimeConfig):
    exp_dir: str = "_tmp/sac"
    env_tag: str = "pend"
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None

    from_pixels: bool = False
    pixels_only: bool = True

    total_timesteps: int = 1000000
    num_envs: int = 1
    frames_per_batch: int = 4
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    replay_size: int = 1000000
    learning_starts: int = 5000
    update_every: int = 1
    updates_per_step: int = 1
    alpha_init: float = 0.2
    target_entropy: float | None = None

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

    @classmethod
    def from_dict(cls, raw: dict) -> "SACConfig":
        data = {k: v for k, v in raw.items() if k in {f.name for f in dataclasses.fields(cls)}}
        for key in [
            "backbone_hidden_sizes",
            "actor_head_hidden_sizes",
            "critic_head_hidden_sizes",
        ]:
            if key in data and data[key] is not None:
                data[key] = tuple(int(x) for x in data[key])
        for key in ["num_envs", "frames_per_batch"]:
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        return cls(**data)


_SAC_RUNTIME_CAPABILITIES = TorchRLRuntimeCapabilities(
    allow_multi_sync_collector=True,
    allow_multi_async_collector=True,
    allow_mps_multi_collectors=False,
    allow_parallel_single_env=True,
)


@dataclasses.dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    last_heldout_return: float | None
    num_steps: int
