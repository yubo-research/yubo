import torch
import torch.nn as nn

from rl.actor_critic import ActionValue, ActorCritic, PolicySpecs
from rl.backbone import (
    BackboneSpec,
    HeadSpec,
    build_backbone,
    build_mlp_head,
    register_backbone,
)


def test_mlp_backbone_shapes():
    spec = BackboneSpec(name="mlp", hidden_sizes=(8, 4), activation="silu", layer_norm=True)
    backbone, out_dim = build_backbone(spec, input_dim=3)
    x = torch.zeros((5, 3))
    y = backbone(x)
    assert y.shape == (5, 4)
    assert out_dim == 4


def test_register_backbone_custom_builder():
    name = "__pytest_identity_backbone__"

    @register_backbone(name)
    def _build_identity(_spec: BackboneSpec, input_dim: int):
        return nn.Identity(), int(input_dim)

    spec = BackboneSpec(name=name, hidden_sizes=(), activation="silu", layer_norm=False)
    backbone, out_dim = build_backbone(spec, input_dim=7)
    x = torch.zeros((3, 7))
    y = backbone(x)
    assert y.shape == (3, 7)
    assert out_dim == 7


def test_build_mlp_head_shapes():
    head = build_mlp_head(HeadSpec(hidden_sizes=(8,), activation="relu"), input_dim=3, output_dim=2)
    x = torch.zeros((4, 3))
    y = head(x)
    assert y.shape == (4, 2)


def test_actorcritic_action_bounds():
    specs = PolicySpecs(
        backbone=BackboneSpec(hidden_sizes=(8,)),
        actor_head=HeadSpec(hidden_sizes=()),
        critic_head=HeadSpec(hidden_sizes=()),
        share_backbone=True,
        log_std_init=0.0,
    )
    agent = ActorCritic(obs_dim=3, act_dim=2, specs=specs)
    obs = torch.zeros((4, 3))
    av = agent.get_action_and_value(obs)
    assert isinstance(av, ActionValue)
    action, logprob, entropy, value = av
    assert action.shape == (4, 2)
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4,)
    assert action.min().item() >= -1.0001
    assert action.max().item() <= 1.0001

    action_mean = agent.act(obs)
    assert action_mean.shape == (4, 2)
    assert action_mean.min().item() >= -1.0001
    assert action_mean.max().item() <= 1.0001

    value2 = agent.get_value(obs)
    assert value2.shape == (4,)

    assert agent.actor_num_params() > 0


def test_actionvalue_namedtuple_fields():
    av = ActionValue(
        action=torch.zeros(2),
        log_prob=torch.ones(1),
        entropy=torch.ones(1) * 2,
        value=torch.ones(1) * 3,
    )
    assert torch.equal(av.action, torch.zeros(2))
    assert torch.equal(av.log_prob, torch.ones(1))
    assert torch.equal(av.entropy, torch.ones(1) * 2)
    assert torch.equal(av.value, torch.ones(1) * 3)
