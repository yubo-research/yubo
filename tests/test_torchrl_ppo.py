from rl.torchrl.common.env_contract import ObservationContract
from rl.torchrl.ppo import core as torchrl_on_policy_core
from rl.torchrl.ppo.core import _TanhNormal as _CoreTanhNormal


def test_ppo_tanhnormal_support_property():
    dist = torchrl_on_policy_core._TanhNormal(
        loc=torchrl_on_policy_core.torch.zeros(1),
        scale=torchrl_on_policy_core.torch.ones(1),
    )
    assert dist.support is torchrl_on_policy_core.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_property():
    dist = _CoreTanhNormal(
        loc=torchrl_on_policy_core.torch.zeros(1),
        scale=torchrl_on_policy_core.torch.ones(1),
    )
    assert dist.support is torchrl_on_policy_core.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_property_module_path():
    dist = torchrl_on_policy_core._TanhNormal(
        loc=torchrl_on_policy_core.torch.zeros(1),
        scale=torchrl_on_policy_core.torch.ones(1),
    )
    assert dist.support is torchrl_on_policy_core.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_fget_direct_call():
    dist = torchrl_on_policy_core._TanhNormal(
        loc=torchrl_on_policy_core.torch.zeros(1),
        scale=torchrl_on_policy_core.torch.ones(1),
    )
    support_value = torchrl_on_policy_core._TanhNormal.support.fget(dist)
    assert support_value is torchrl_on_policy_core.torch.distributions.constraints.real


def test_discrete_actor_net_handles_unbatched_atari_obs():
    ppo = torchrl_on_policy_core
    backbone = ppo.op_deps.backbone.build_backbone(
        ppo.op_deps.backbone.BackboneSpec(
            name="nature_cnn_atari",
            hidden_sizes=(),
            activation="relu",
            layer_norm=False,
        ),
        input_dim=64,
    )[0]
    head = ppo.op_deps.backbone.build_mlp_head(
        ppo.op_deps.backbone.HeadSpec(hidden_sizes=(), activation="relu"),
        input_dim=64,
        output_dim=6,
    )
    net = ppo._DiscreteActorNet(
        backbone,
        head,
        ppo.op_deps.torchrl_common.ObsScaler(None, None),
        obs_contract=ObservationContract(mode="pixels", raw_shape=(4, 84, 84), model_channels=4, image_size=84),
    )
    obs = ppo.torch.randint(0, 256, (4, 84, 84), dtype=ppo.torch.uint8).float() / 255.0
    logits = net(obs)
    assert logits.shape == (6,)


def test_resolve_backbone_name_for_pixel_contract():
    ppo = torchrl_on_policy_core
    assert (
        ppo.op_deps.torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="pixels", raw_shape=(4, 84, 84, 1), model_channels=4, image_size=84),
        )
        == "nature_cnn_atari"
    )
    assert (
        ppo.op_deps.torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="pixels", raw_shape=(84, 84, 3), model_channels=3, image_size=84),
        )
        == "nature_cnn"
    )
    assert (
        ppo.op_deps.torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="vector", raw_shape=(17,), vector_dim=17),
        )
        == "mlp"
    )
