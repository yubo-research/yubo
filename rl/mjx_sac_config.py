from __future__ import annotations

import dataclasses

from rl.core.grouped_config import dataclass_field_names, parse_dataclass_section
from rl.mjx_sac_config_sections import (
    MJXSACCollectorConfig,
    MJXSACLossConfig,
    MJXSACOptimConfig,
)


@dataclasses.dataclass
class MJXSACSections:
    collector: MJXSACCollectorConfig = dataclasses.field(default_factory=MJXSACCollectorConfig)
    optim: MJXSACOptimConfig = dataclasses.field(default_factory=MJXSACOptimConfig)
    loss: MJXSACLossConfig = dataclasses.field(default_factory=MJXSACLossConfig)


@dataclasses.dataclass
class MJXSACConfig:
    exp_dir: str = "runs/rl/mjx_sac"
    env_tag: str = "mujoco_playground:CheetahRun"
    seed: int = 0
    hidden_size: int = 256
    log_interval: int = 10
    sections: MJXSACSections = dataclasses.field(default_factory=MJXSACSections)

    @classmethod
    def from_dict(cls, raw: dict) -> "MJXSACConfig":
        grouped = {"collector", "optim", "loss"}
        root_fields = dataclass_field_names(cls) - {"sections"}
        unknown = sorted(set(raw) - root_fields - grouped)
        if unknown:
            raise ValueError(f"Unknown MJX SAC config fields: {', '.join(unknown)}.")
        data = {key: raw[key] for key in root_fields if key in raw}
        data["sections"] = MJXSACSections(
            collector=parse_dataclass_section(raw, "collector", MJXSACCollectorConfig, label="MJX SAC"),
            optim=parse_dataclass_section(raw, "optim", MJXSACOptimConfig, label="MJX SAC"),
            loss=parse_dataclass_section(raw, "loss", MJXSACLossConfig, label="MJX SAC"),
        )
        return cls(**data)

    @property
    def collector(self) -> MJXSACCollectorConfig:
        return self.sections.collector

    @property
    def optim(self) -> MJXSACOptimConfig:
        return self.sections.optim

    @property
    def loss(self) -> MJXSACLossConfig:
        return self.sections.loss

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
