from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.backends.torchrl.common import env_contract as torchrl_env_contract
from rl.policy_backbone import (
    ActorBackbonePolicy,
    ActorBackbonePolicyFactory,
    ActorPolicySpec,
    DiscreteActorBackbonePolicy,
    DiscreteActorBackbonePolicyFactory,
    DiscreteActorPolicySpec,
    GaussianActorBackbonePolicy,
)


def test_actor_policy_spec():
    spec = ActorPolicySpec(
        backbone=BackboneSpec(hidden_sizes=(4,)),
        head=HeadSpec(),
        action_squash=True,
        param_scale=0.5,
    )
    assert spec.action_squash is True
    assert spec.param_scale == 0.5


def test_actor_backbone_policy():
    gym_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(4,)))
    env_conf = SimpleNamespace(
        gym_conf=gym_conf,
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    spec = ActorPolicySpec(
        backbone=BackboneSpec(hidden_sizes=(8,)),
        head=HeadSpec(),
    )
    policy = ActorBackbonePolicy(env_conf, spec)
    out = policy(torch.randn(4))
    assert out.shape == (2,)
    assert -1.01 <= out.min() <= out.max() <= 1.01


def test_actor_backbone_policy_factory():
    factory = ActorBackbonePolicyFactory(
        backbone=BackboneSpec(hidden_sizes=(4,)),
        head=HeadSpec(),
    )
    gym_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(6,)))
    env_conf = SimpleNamespace(
        gym_conf=gym_conf,
        action_space=SimpleNamespace(shape=(1,)),
        ensure_spaces=lambda: None,
    )
    policy = factory(env_conf)
    assert isinstance(policy, ActorBackbonePolicy)


def test_gaussian_actor_backbone_policy():
    gym_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(4,)))
    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=gym_conf,
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    policy = GaussianActorBackbonePolicy(env_conf, variant="rl-gauss-tanh")
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action.shape == (2,)
    assert -1.01 <= action.min() <= action.max() <= 1.01
    assert policy.num_params() > 0
    p = policy.get_params()
    policy.set_params(p)
    np.testing.assert_allclose(policy.get_params(), p)


def test_get_env_conf_gauss_uses_gaussian_actor_backbone():
    import problems.env_conf_atari_dm  # noqa: F401 - register atari/dm handlers
    from problems.env_conf import default_policy, get_env_conf

    ec = get_env_conf("dm:cheetah-run:gauss", problem_seed=0)
    policy = default_policy(ec)
    assert isinstance(policy, GaussianActorBackbonePolicy)


def _fake_atari_env_conf():
    gym_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(4, 84, 84)))
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=gym_conf,
        action_space=gym.spaces.Discrete(6),
        from_pixels=True,
        ensure_spaces=lambda: None,
    )


def _atari_discrete_spec():
    return DiscreteActorPolicySpec(
        backbone=BackboneSpec(
            name="mlp",
            hidden_sizes=(16, 16),
            activation="relu",
            layer_norm=False,
        ),
        head=HeadSpec(hidden_sizes=(16, 16), activation="relu"),
        param_scale=0.5,
    )


def test_discrete_actor_backbone_policy_pixels_atari():
    env_conf = _fake_atari_env_conf()
    policy = DiscreteActorBackbonePolicy(env_conf, _atari_discrete_spec())
    obs = torch.rand(4, 84, 84)
    logits = policy.forward(obs)
    assert logits.shape == (6,)
    action = policy(obs.numpy())
    assert isinstance(action, int)
    assert 0 <= action < 6
    params = policy.get_params()
    policy.set_params(params)
    np.testing.assert_allclose(policy.get_params(), params)


def test_discrete_actor_backbone_policy_param_count_matches_ppo_build_path():
    env_conf = _fake_atari_env_conf()
    spec = _atari_discrete_spec()
    policy = DiscreteActorBackbonePolicy(env_conf, spec)

    obs_contract = torchrl_env_contract.resolve_observation_contract(env_conf, default_image_size=84)
    backbone_name = torchrl_env_contract.resolve_backbone_name(spec.backbone.name, obs_contract)
    backbone_spec = BackboneSpec(
        name=backbone_name,
        hidden_sizes=spec.backbone.hidden_sizes,
        activation=spec.backbone.activation,
        layer_norm=spec.backbone.layer_norm,
    )
    ppo_backbone, feat_dim = build_backbone(backbone_spec, input_dim=64)
    ppo_head = build_mlp_head(spec.head, input_dim=feat_dim, output_dim=env_conf.action_space.n)
    ppo_param_count = sum(p.numel() for p in ppo_backbone.parameters()) + sum(p.numel() for p in ppo_head.parameters())
    assert policy.num_params() == ppo_param_count


def test_get_env_conf_atari_mlp16_uses_discrete_actor_backbone_policy():
    import problems.env_conf as env_conf_module

    class _FakeAtariDM:
        @staticmethod
        def get_atari_parsers_and_factories():
            def _parse_atari_tag(_tag):
                return "ALE/Pong-v5"

            return _parse_atari_tag, object, object, object

    old_module = env_conf_module._atari_dm_module
    env_conf_module.register_atari_dm(_FakeAtariDM())
    try:
        ec = env_conf_module.get_env_conf("atari:Pong:mlp16", problem_seed=0)
        policy = ec.policy_class(_fake_atari_env_conf())
    finally:
        env_conf_module.register_atari_dm(old_module)
    assert isinstance(policy, DiscreteActorBackbonePolicy)


def test_discrete_actor_backbone_policy_factory():
    factory = DiscreteActorBackbonePolicyFactory(
        backbone=BackboneSpec(
            name="mlp",
            hidden_sizes=(16, 16),
            activation="relu",
            layer_norm=False,
        ),
        head=HeadSpec(hidden_sizes=(16, 16), activation="relu"),
    )
    policy = factory(_fake_atari_env_conf())
    assert isinstance(policy, DiscreteActorBackbonePolicy)
