"""PufferLib R2D2 configuration."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class R2D2Config:
    exp_dir: str = "_tmp/rl/atari/r2d2"
    env_tag: str = "atari:MontezumaRevenge"
    seed: int = 1
    problem_seed: int | None = None

    total_timesteps: int = 1000000
    learning_rate: float = 1e-4
    gamma: float = 0.997
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200000
    num_envs: int = 16
    unroll_length: int = 80
    burn_in: int = 40
    recurrent_hidden_dim: int = 256
    replay_capacity: int = 500000
    batch_size: int = 64
    learning_starts: int = 20000
    updates_per_step: int = 1
    target_update_interval: int = 2500
    value_rescaling: bool = True

    device: str = "auto"
    eval_interval: int = 20000
    eval_episodes: int = 3
    log_interval: int = 5000
    checkpoint_interval: int | None = 100

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict):
        data = dict(raw)
        for key in [
            "seed",
            "problem_seed",
            "total_timesteps",
            "eps_decay_steps",
            "num_envs",
            "unroll_length",
            "burn_in",
            "recurrent_hidden_dim",
            "replay_capacity",
            "batch_size",
            "learning_starts",
            "updates_per_step",
            "target_update_interval",
            "eval_interval",
            "eval_episodes",
            "log_interval",
        ]:
            if key in data:
                data[key] = int(data[key])
        for key in ["learning_rate", "gamma", "eps_start", "eps_end"]:
            if key in data:
                data[key] = float(data[key])
        if data.get("checkpoint_interval") is not None:
            data["checkpoint_interval"] = int(data["checkpoint_interval"])
        return cls(**data)
