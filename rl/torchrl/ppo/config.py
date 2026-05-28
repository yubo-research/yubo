from __future__ import annotations

import dataclasses

from rl.config_model_defaults import (
    apply_ppo_env_model_defaults,
    reject_model_config_keys,
)
from rl.core.grouped_config import dataclass_field_names, parse_dataclass_section
from rl.core.torchrl_runtime import TorchRLRuntimeCapabilities, resolve_torchrl_runtime
from rl.core.torchrl_runtime_dtos import TorchRLRuntime, TorchRLRuntimeRequest
from rl.core.torchrl_runtime_request import make_torchrl_runtime_request

from .config_collector import PPOCollectorConfig, PPOLossConfig, PPOOptimConfig
from .config_env import PPOEnvConfig
from .config_run import PPOCheckpointConfig, PPOEvalConfig, PPOProfileConfig


@dataclasses.dataclass
class PPOConfig:
    exp_dir: str = "_tmp/ppo"
    env_tag: str = "pend"
    policy_tag: str | None = None
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None
    device: str = "auto"
    from_pixels: bool = False
    pixels_only: bool = True
    log_interval: int = 1
    env: PPOEnvConfig = dataclasses.field(default_factory=PPOEnvConfig)
    collector: PPOCollectorConfig = dataclasses.field(default_factory=PPOCollectorConfig)
    optim: PPOOptimConfig = dataclasses.field(default_factory=PPOOptimConfig)
    loss: PPOLossConfig = dataclasses.field(default_factory=PPOLossConfig)
    eval: PPOEvalConfig = dataclasses.field(default_factory=PPOEvalConfig)
    checkpoint: PPOCheckpointConfig = dataclasses.field(default_factory=PPOCheckpointConfig)
    profile: PPOProfileConfig = dataclasses.field(default_factory=PPOProfileConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "PPOConfig":
        reject_model_config_keys(raw, algo="ppo")
        apply_ppo_env_model_defaults(raw)
        sections = {
            "env",
            "collector",
            "optim",
            "loss",
            "eval",
            "checkpoint",
            "profile",
        }
        root_fields = dataclass_field_names(cls) - sections
        unknown = sorted(set(raw) - root_fields - sections)
        if unknown:
            raise ValueError(f"Unknown PPO config fields: {', '.join(unknown)}. Use grouped PPO tables.")
        data = {key: raw[key] for key in root_fields if key in raw}
        data.update(
            env=parse_dataclass_section(raw, "env", PPOEnvConfig, label="PPO"),
            collector=parse_dataclass_section(raw, "collector", PPOCollectorConfig, label="PPO"),
            optim=parse_dataclass_section(raw, "optim", PPOOptimConfig, label="PPO"),
            loss=parse_dataclass_section(raw, "loss", PPOLossConfig, label="PPO"),
            eval=parse_dataclass_section(raw, "eval", PPOEvalConfig, label="PPO"),
            checkpoint=parse_dataclass_section(raw, "checkpoint", PPOCheckpointConfig, label="PPO"),
            profile=parse_dataclass_section(raw, "profile", PPOProfileConfig, label="PPO"),
        )
        return cls(**data)

    def runtime_num_envs(self) -> int:
        return int(self.collector.num_envs)

    def runtime_request(self) -> TorchRLRuntimeRequest:
        return make_torchrl_runtime_request(
            env_tag=self.env_tag,
            device=self.device,
            collector_backend=str(self.collector.backend),
            single_env_backend=str(self.collector.single_env_backend),
            num_envs=int(self.runtime_num_envs()),
            collector_workers=self.collector.workers,
        )

    def resolve_runtime(self, *, capabilities: TorchRLRuntimeCapabilities | None = None) -> TorchRLRuntime:
        resolved_capabilities = capabilities if capabilities is not None else TorchRLRuntimeCapabilities()
        return resolve_torchrl_runtime(self.runtime_request(), capabilities=resolved_capabilities)


_PPO_RUNTIME_CAPABILITIES = TorchRLRuntimeCapabilities(
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
    num_iterations: int
