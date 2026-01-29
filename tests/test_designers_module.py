import numpy as np
import pytest


class MockPolicy:
    def __init__(self, num_params=5):
        self._num_params = num_params
        self.problem_seed = 0

    def num_params(self):
        return self._num_params

    def set_params(self, x):
        pass

    def get_params(self):
        return np.zeros(self._num_params)

    def clone(self):
        return MockPolicy(self._num_params)


def test_designers_init():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    assert designers is not None


def test_designers_create_random():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("random")
    assert designer is not None


def test_designers_create_sobol():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("sobol")
    assert designer is not None


def test_designers_create_lhd():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("lhd")
    assert designer is not None


def test_designers_create_center():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("center")
    assert designer is not None


def test_designers_no_such_designer():
    from optimizer.designers import Designers, NoSuchDesignerError

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    with pytest.raises(NoSuchDesignerError):
        designers.create("nonexistent_designer")


def test_designers_is_valid_raises():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    # is_valid uses self._designers which is not defined - this should raise AttributeError
    with pytest.raises(AttributeError):
        designers.is_valid("random")
