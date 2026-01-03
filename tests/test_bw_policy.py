import numpy as np
import pytest


class _Shape:
    def __init__(self, shape):
        self.shape = shape


class _DummyGymConf:
    def __init__(self, num_state):
        self.state_space = _Shape((num_state,))


class _DummyEnvConf:
    def __init__(self, num_state=24, num_action=4):
        self.env_name = "BipedalWalker-v3"
        self.problem_seed = 0
        self.gym_conf = _DummyGymConf(num_state)
        self.action_space = _Shape((num_action,))


def test_bw_policy_shapes_and_bounds():
    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    env_conf = _DummyEnvConf()
    p = BipedalWalkerPolicy(env_conf)
    assert p.num_params() == 16

    s = np.zeros(24, dtype=np.float64)
    a = p(s)
    assert a.shape == (4,)
    assert np.all(np.isfinite(a))
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_bw_policy_set_get_and_clone():
    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    env_conf = _DummyEnvConf()
    p = BipedalWalkerPolicy(env_conf)
    x = np.linspace(-1.0, 1.0, p.num_params())
    p.set_params(x)
    assert np.allclose(p.get_params(), x)

    s = np.zeros(24, dtype=np.float64)
    a0 = p(s)
    assert a0.shape == (4,)

    p2 = p.clone()
    assert np.allclose(p2.get_params(), x)

    p3 = BipedalWalkerPolicy(env_conf)
    p3.set_params(x)
    a2 = p2(s)
    a3 = p3(s)
    assert np.allclose(a2, a3)


def test_bw_policy_set_params_range_check():
    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    env_conf = _DummyEnvConf()
    p = BipedalWalkerPolicy(env_conf)
    x = np.zeros(p.num_params(), dtype=np.float64)
    x[0] = 2.0
    with pytest.raises(AssertionError):
        p.set_params(x)


def test_bw_policy_contact_indices_drive_stance_side():
    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    env_conf = _DummyEnvConf()
    p = BipedalWalkerPolicy(env_conf)
    x = np.zeros(p.num_params(), dtype=np.float64)
    p.set_params(x)

    s = np.zeros(24, dtype=np.float64)
    s[0] = 0.0
    s[1] = 0.0
    s[2] = 0.0
    s[3] = 0.0

    s_left_contact = s.copy()
    s_left_contact[8] = 1.0
    s_left_contact[13] = 0.0
    a_left = p(s_left_contact)
    assert a_left.shape == (4,)

    s_right_contact = s.copy()
    s_right_contact[8] = 0.0
    s_right_contact[13] = 1.0
    a_right = p(s_right_contact)
    assert a_right.shape == (4,)

    assert not np.allclose(a_left, a_right)


def test_bw_feat_policy_shapes_and_bounds():
    from problems.bipedal_walker_feat_policy import BipedalWalkerFeatPolicy

    env_conf = _DummyEnvConf()
    p = BipedalWalkerFeatPolicy(env_conf)
    assert p.num_params() == 69

    s = np.zeros(24, dtype=np.float64)
    a = p(s)
    assert a.shape == (4,)
    assert np.all(np.isfinite(a))
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_linear_policy_default_is_callable():
    from problems.linear_policy import LinearPolicy

    env_conf = _DummyEnvConf(num_state=24, num_action=4)
    p = LinearPolicy(env_conf)
    s = np.zeros(24, dtype=np.float64)
    a = p(s)
    assert a.shape == (4,)
    assert np.all(np.isfinite(a))
    assert a.min() >= -1.0 and a.max() <= 1.0
