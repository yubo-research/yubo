import numpy as np
import torch

from problems.mlp_torch_policy import MLPPolicyModule
from problems.torch_policy import TorchPolicy


class _FakeGymConf:
    def __init__(self, num_state):
        self.state_space = type("S", (), {"shape": (num_state,)})()
        self.transform_state = True


class _FakeEnvConf:
    def __init__(self, num_state):
        self.problem_seed = 0
        self.gym_conf = _FakeGymConf(num_state)


def test_forward_returns_correct_shape():
    module = MLPPolicyModule(8, 3)
    x = torch.randn(8)
    y = module(x)
    assert y.shape == (3,)


def test_custom_hidden_sizes():
    module = MLPPolicyModule(4, 2, hidden_sizes=(64, 32, 16))
    x = torch.randn(4)
    y = module(x)
    assert y.shape == (2,)


def test_works_with_torch_policy():
    module = MLPPolicyModule(4, 2)
    policy = TorchPolicy(module, _FakeEnvConf(4))
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action.shape == (2,)
    assert action.min() >= -1.0
    assert action.max() <= 1.0
