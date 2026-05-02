from types import SimpleNamespace

import pytest

from policies.env_utils import get_action_space, get_obs_act_dims, get_obs_space


def test_get_obs_space_from_state_space():
    env_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(4,)))
    obs = get_obs_space(env_conf)
    assert obs.shape == (4,)


def test_get_obs_space_from_gym_conf():
    env_conf = SimpleNamespace(
        state_space=None,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(8,))),
    )
    obs = get_obs_space(env_conf)
    assert obs.shape == (8,)


def test_get_obs_space_missing_raises():
    env_conf = SimpleNamespace(state_space=None, gym_conf=None)
    with pytest.raises(ValueError, match="Observation space not found"):
        get_obs_space(env_conf)


def test_get_action_space():
    env_conf = SimpleNamespace(action_space=SimpleNamespace(shape=(2,)))
    act = get_action_space(env_conf)
    assert act.shape == (2,)


def test_get_action_space_missing_raises():
    env_conf = SimpleNamespace(action_space=None)
    with pytest.raises(ValueError, match="Action space not found"):
        get_action_space(env_conf)


def test_get_obs_act_dims():
    env_conf = SimpleNamespace(
        state_space=SimpleNamespace(shape=(4,)),
        action_space=SimpleNamespace(shape=(2,)),
    )
    obs_dim, act_dim = get_obs_act_dims(env_conf)
    assert obs_dim == 4
    assert act_dim == 2
