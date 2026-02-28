"""TorchRL TD-MPC2 configuration."""

from __future__ import annotations

import dataclasses

from rl.torchrl.common.runtime import TorchRLRuntimeCapabilities, TorchRLRuntimeConfig


@dataclasses.dataclass
class TDMPC2Config(TorchRLRuntimeConfig):
    exp_dir: str = "_tmp/rl/dm_control/tdmpc2"
    env_tag: str = "dm:humanoid-run"
    seed: int = 1
    problem_seed: int | None = None

    total_timesteps: int = 1000000
    gamma: float = 0.99
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    tau: float = 0.01
    horizon: int = 5
    rollout_batch_size: int = 256
    latent_dim: int = 256
    hidden_dim: int = 512
    replay_capacity: int = 200000
    warmup_steps: int = 2000
    updates_per_step: int = 1
    plan_samples: int = 128
    plan_elites: int = 16
    plan_iters: int = 4
    exploration_std: float = 0.15

    eval_interval: int = 5000
    eval_episodes: int = 3
    log_interval: int = 1000
    checkpoint_interval: int | None = 100

    def runtime_num_envs(self) -> int:
        return 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict):
        data = dict(raw)
        for key in [
            "seed",
            "problem_seed",
            "total_timesteps",
            "horizon",
            "rollout_batch_size",
            "latent_dim",
            "hidden_dim",
            "replay_capacity",
            "warmup_steps",
            "updates_per_step",
            "plan_samples",
            "plan_elites",
            "plan_iters",
            "eval_interval",
            "eval_episodes",
            "log_interval",
        ]:
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        for key in [
            "gamma",
            "model_lr",
            "actor_lr",
            "value_lr",
            "tau",
            "exploration_std",
        ]:
            if key in data and data[key] is not None:
                data[key] = float(data[key])
        if data.get("checkpoint_interval") is not None:
            data["checkpoint_interval"] = int(data["checkpoint_interval"])
        return cls(**data)


_TDMPC2_RUNTIME_CAPABILITIES = TorchRLRuntimeCapabilities(
    allow_multi_sync_collector=False,
    allow_multi_async_collector=False,
    allow_mps_multi_collectors=False,
    allow_parallel_single_env=False,
    allow_mps_parallel_single_env=False,
)
