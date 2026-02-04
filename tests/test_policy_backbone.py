import numpy as np

from problems.env_conf import get_env_conf
from rl.backbone import BackboneSpec, HeadSpec
from rl.policy_backbone import ActorBackbonePolicy, ActorBackbonePolicyFactory, ActorPolicySpec


def test_actor_backbone_policy_roundtrip():
    env_conf = get_env_conf("pend", problem_seed=0, noise_seed_0=0)
    env_conf.ensure_spaces()
    spec = BackboneSpec(hidden_sizes=(8,), activation="silu", layer_norm=True)
    head = HeadSpec(hidden_sizes=(), activation="silu")
    policy = ActorBackbonePolicy(
        env_conf,
        spec=ActorPolicySpec(
            backbone=spec,
            head=head,
            action_squash=True,
            param_scale=0.5,
        ),
    )
    theta = policy.get_params()
    new_theta = np.zeros_like(theta)
    policy.set_params(new_theta)
    theta2 = policy.get_params()
    assert np.allclose(theta2, new_theta)


def test_actor_backbone_policy_action_bounds():
    env_conf = get_env_conf("pend", problem_seed=1, noise_seed_0=0)
    env_conf.ensure_spaces()
    spec = BackboneSpec(hidden_sizes=(8,), activation="silu", layer_norm=True)
    head = HeadSpec(hidden_sizes=(), activation="silu")
    policy = ActorBackbonePolicy(
        env_conf,
        spec=ActorPolicySpec(
            backbone=spec,
            head=head,
            action_squash=True,
            param_scale=0.5,
        ),
    )
    obs = np.zeros(env_conf.gym_conf.state_space.shape, dtype=np.float32)
    action = policy(obs)
    assert action.min() >= -1.0001
    assert action.max() <= 1.0001


def test_actor_backbone_policy_factory_builds_policy():
    env_conf = get_env_conf("pend", problem_seed=2, noise_seed_0=0)
    env_conf.ensure_spaces()
    spec = BackboneSpec(hidden_sizes=(8,), activation="silu", layer_norm=True)
    head = HeadSpec(hidden_sizes=(), activation="silu")
    factory = ActorBackbonePolicyFactory(
        backbone=spec,
        head=head,
        action_squash=True,
        param_scale=0.5,
    )
    policy = factory(env_conf)
    assert isinstance(policy, ActorBackbonePolicy)
