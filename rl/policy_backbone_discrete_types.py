from __future__ import annotations

from dataclasses import dataclass

from rl.backbone import BackboneSpec, HeadSpec


@dataclass
class DiscreteActorPolicySpec:
    backbone: BackboneSpec
    head: HeadSpec
    param_scale: float = 0.5


__all__ = ["DiscreteActorPolicySpec"]
