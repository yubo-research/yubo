import numpy as np
import torch.nn as nn

from problems.torch_policy import TorchPolicy


class _FakeGymConf:
    def __init__(self, num_state):
        self.state_space = type("S", (), {"shape": (num_state,)})()
        self.transform_state = True


class _FakeEnvConf:
    def __init__(self, num_state):
        self.problem_seed = 0
        self.gym_conf = _FakeGymConf(num_state)


class _FakeEnvConfNoGym:
    problem_seed = 0
    gym_conf = None


def test_call_returns_numpy():
    module = nn.Linear(4, 2)
    policy = TorchPolicy(module, _FakeEnvConf(4))
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)


def test_output_clamped():
    module = nn.Linear(4, 2)
    # Set large weights to force large outputs
    nn.init.constant_(module.weight, 100.0)
    nn.init.constant_(module.bias, 100.0)
    policy = TorchPolicy(module, _FakeEnvConf(4))
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action.max() <= 1.0
    assert action.min() >= -1.0


def test_no_gym_conf_skips_normalization_and_clamping():
    module = nn.Linear(4, 2)
    nn.init.constant_(module.weight, 100.0)
    nn.init.constant_(module.bias, 100.0)
    policy = TorchPolicy(module, _FakeEnvConfNoGym())

    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    # Without clamping, large weights produce outputs >> 1
    assert action.max() > 1.0


def test_no_gym_conf_handles_multidim_state():
    """TorchPolicy with gym_conf=None should handle non-1D states (e.g. images)."""
    # Simple conv module that takes (batch, 1, 4, 4) input
    module = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))
    policy = TorchPolicy(module, _FakeEnvConfNoGym())

    state = np.random.rand(2, 1, 4, 4).astype(np.float32)
    action = policy(state)
    assert action.shape == (2, 3)
