from __future__ import annotations

import dataclasses

from rl.config_model_defaults import apply_sac_env_model_defaults, reject_model_config_keys
from rl.core.grouped_config import dataclass_field_names, parse_dataclass_section
from rl.core.torchrl_runtime import TorchRLRuntimeCapabilities, resolve_torchrl_runtime
from rl.core.torchrl_runtime_dtos import TorchRLRuntime, TorchRLRuntimeRequest
from rl.core.torchrl_runtime_request import make_torchrl_runtime_request

from .config_collector import SACCollectorConfig, SACOptimConfig, SACReplayBufferConfig
from .config_objective import SACEvalConfig, SACLossConfig, SACTargetNetUpdaterConfig
from .config_run import SACCheckpointConfig


@dataclasses.dataclass
class SACConfig:
    exp_dir: str = "_tmp/sac"
    env_tag: str = "pend"
    policy_tag: str | None = None
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None
    device: str = "auto"
    from_pixels: bool = False
    pixels_only: bool = True
    log_interval_steps: int = 1000
    collector: SACCollectorConfig = dataclasses.field(default_factory=SACCollectorConfig)
    replay_buffer: SACReplayBufferConfig = dataclasses.field(default_factory=SACReplayBufferConfig)
    optim: SACOptimConfig = dataclasses.field(default_factory=SACOptimConfig)
    loss: SACLossConfig = dataclasses.field(default_factory=SACLossConfig)
    target_net_updater: SACTargetNetUpdaterConfig = dataclasses.field(default_factory=SACTargetNetUpdaterConfig)
    eval: SACEvalConfig = dataclasses.field(default_factory=SACEvalConfig)
    checkpoint: SACCheckpointConfig = dataclasses.field(default_factory=SACCheckpointConfig)

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

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "SACConfig":
        reject_model_config_keys(raw, algo="sac")
        apply_sac_env_model_defaults(raw)
        sections = {"collector", "replay_buffer", "optim", "loss", "target_net_updater", "eval", "checkpoint"}
        root_fields = dataclass_field_names(cls) - sections
        unknown = sorted(set(raw) - root_fields - sections)
        if unknown:
            raise ValueError(f"Unknown SAC config fields: {', '.join(unknown)}. Use grouped SAC tables.")
        data = {key: raw[key] for key in root_fields if key in raw}
        data.update(
            collector=parse_dataclass_section(raw, "collector", SACCollectorConfig, label="SAC"),
            replay_buffer=parse_dataclass_section(raw, "replay_buffer", SACReplayBufferConfig, label="SAC"),
            optim=parse_dataclass_section(raw, "optim", SACOptimConfig, label="SAC"),
            loss=parse_dataclass_section(raw, "loss", SACLossConfig, label="SAC"),
            target_net_updater=parse_dataclass_section(raw, "target_net_updater", SACTargetNetUpdaterConfig, label="SAC"),
            eval=parse_dataclass_section(raw, "eval", SACEvalConfig, label="SAC"),
            checkpoint=parse_dataclass_section(raw, "checkpoint", SACCheckpointConfig, label="SAC"),
        )
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
