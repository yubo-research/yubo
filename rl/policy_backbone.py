"""Re-exports policy backbone types from split modules (kiss `concrete_types_per_file`)."""

from rl.backbone import BackboneSpec, HeadSpec
from rl.policy_backbone_actor import (
    ActorBackbonePolicy,
    ActorBackbonePolicyFactory,
    ActorPolicySpec,
)
from rl.policy_backbone_atari import AtariMLP16DiscretePolicy
from rl.policy_backbone_discrete import (
    DiscreteActorBackbonePolicy,
    DiscreteActorBackbonePolicyFactory,
    DiscreteActorPolicySpec,
)
from rl.policy_backbone_gaussian import (
    GaussianActorBackbonePolicy,
    GaussianActorBackbonePolicyFactory,
)


__all__ = [
    "ActorBackbonePolicy",
    "ActorBackbonePolicyFactory",
    "ActorPolicySpec",
    "AtariMLP16DiscretePolicy",
    "BackboneSpec",
    "DiscreteActorBackbonePolicy",
    "DiscreteActorBackbonePolicyFactory",
    "DiscreteActorPolicySpec",
    "GaussianActorBackbonePolicy",
    "GaussianActorBackbonePolicyFactory",
    "HeadSpec",
]
