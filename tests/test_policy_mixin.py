import numpy as np
import torch.nn as nn

from problems.policy_mixin import PolicyParamsMixin


def test_policy_params_mixin_class_exists():
    assert PolicyParamsMixin is not None
    assert hasattr(PolicyParamsMixin, "num_params")
    assert hasattr(PolicyParamsMixin, "get_params")
    assert hasattr(PolicyParamsMixin, "set_params")
    assert hasattr(PolicyParamsMixin, "clone")


class MockPolicy(PolicyParamsMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self._flat_params_init = np.zeros(9, dtype=np.float32)  # 2*3 + 3 = 9 params
        self._const_scale = 1.0

    def reset_state(self):
        pass


def test_policy_params_mixin_num_params():
    policy = MockPolicy()
    assert policy.num_params() == 9


def test_policy_params_mixin_get_set_params():
    policy = MockPolicy()
    params = policy.get_params()
    assert params.shape == (9,)

    new_params = np.zeros(9, dtype=np.float32)
    policy.set_params(new_params)
    params2 = policy.get_params()
    assert np.allclose(params2, new_params)


def test_policy_params_mixin_clone():
    policy = MockPolicy()
    clone = policy.clone()
    assert clone is not policy
    assert clone.num_params() == policy.num_params()
