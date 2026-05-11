from rl.backbone import BackboneSpec, HeadSpec

from .discrete import DiscreteActorBackbonePolicy, DiscreteActorPolicySpec


class AtariMLP16DiscretePolicy(DiscreteActorBackbonePolicy):
    def __init__(self, env_conf):
        super().__init__(
            env_conf,
            DiscreteActorPolicySpec(
                backbone=BackboneSpec(
                    name="mlp",
                    hidden_sizes=(16, 16),
                    activation="relu",
                    layer_norm=False,
                ),
                head=HeadSpec(hidden_sizes=(16, 16), activation="relu"),
                param_scale=0.5,
            ),
        )
