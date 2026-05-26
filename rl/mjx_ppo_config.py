from __future__ import annotations

import dataclasses

from rl.core.grouped_config import dataclass_field_names, parse_dataclass_section
from rl.mjx_ppo_config_sections import MJXPPOCollectorConfig, MJXPPOLossConfig, MJXPPOOptimConfig


@dataclasses.dataclass
class MJXPPOSections:
    collector: MJXPPOCollectorConfig = dataclasses.field(default_factory=MJXPPOCollectorConfig)
    optim: MJXPPOOptimConfig = dataclasses.field(default_factory=MJXPPOOptimConfig)
    loss: MJXPPOLossConfig = dataclasses.field(default_factory=MJXPPOLossConfig)


@dataclasses.dataclass
class MJXPPOConfig:
    exp_dir: str = "runs/rl/mjx_ppo"
    env_tag: str = "brax:ant"
    seed: int = 0
    hidden_size: int = 64
    log_interval: int = 10
    sections: MJXPPOSections = dataclasses.field(default_factory=MJXPPOSections)

    @classmethod
    def from_dict(cls, raw: dict) -> "MJXPPOConfig":
        grouped = {"collector", "optim", "loss"}
        root_fields = dataclass_field_names(cls) - {"sections"}
        unknown = sorted(set(raw) - root_fields - grouped)
        if unknown:
            raise ValueError(f"Unknown MJX PPO config fields: {', '.join(unknown)}.")
        data = {key: raw[key] for key in root_fields if key in raw}
        data["sections"] = MJXPPOSections(
            collector=parse_dataclass_section(raw, "collector", MJXPPOCollectorConfig, label="MJX PPO"),
            optim=parse_dataclass_section(raw, "optim", MJXPPOOptimConfig, label="MJX PPO"),
            loss=parse_dataclass_section(raw, "loss", MJXPPOLossConfig, label="MJX PPO"),
        )
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def collector(self) -> MJXPPOCollectorConfig:
        return self.sections.collector

    @property
    def optim(self) -> MJXPPOOptimConfig:
        return self.sections.optim

    @property
    def loss(self) -> MJXPPOLossConfig:
        return self.sections.loss
