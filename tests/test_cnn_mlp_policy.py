"""Tests for CNNMLPPolicy and CNNMLPPolicyFactory."""

import numpy as np
import torch

from problems.cnn_mlp_policy import CNNMLPPolicy, CNNMLPPolicyFactory


class _FakeGymConf:
    def __init__(self):
        self.state_space = type("S", (), {"shape": (84, 84, 3)})()
        self.transform_state = False


class _FakeEnvConf:
    def __init__(self):
        self.problem_seed = 42
        self.gym_conf = _FakeGymConf()
        self.action_space = type("A", (), {"shape": (6,)})()


def test_cnn_mlp_policy_factory():
    """CNNMLPPolicyFactory creates CNNMLPPolicy from env_conf."""
    factory = CNNMLPPolicyFactory((32, 16))
    env_conf = _FakeEnvConf()
    policy = factory(env_conf)
    assert isinstance(policy, CNNMLPPolicy)
    assert policy.problem_seed == 42


def test_cnn_mlp_policy_forward():
    """CNNMLPPolicy forward returns action of correct shape."""
    env_conf = _FakeEnvConf()
    policy = CNNMLPPolicy(env_conf, (32, 16))
    # (N, C, H, W)
    x = torch.rand(2, 3, 84, 84)
    out = policy.forward(x)
    assert out.shape == (2, 6)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_cnn_mlp_policy_call():
    """CNNMLPPolicy __call__ accepts numpy HWC and returns action."""
    env_conf = _FakeEnvConf()
    policy = CNNMLPPolicy(env_conf, (32, 16))
    state = np.random.rand(84, 84, 3).astype(np.float32)
    action = policy(state)
    assert action.shape == (6,)
    assert action.min() >= -1.0 and action.max() <= 1.0


def test_cnn_mlp_policy_num_params():
    """CNNMLPPolicy has num_params from PolicyParamsMixin."""
    env_conf = _FakeEnvConf()
    policy = CNNMLPPolicy(env_conf, (32, 16))
    n = policy.num_params()
    assert n > 0
    assert n == sum(p.numel() for p in policy.parameters())
