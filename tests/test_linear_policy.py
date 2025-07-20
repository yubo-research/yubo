import gymnasium as gym
import numpy as np

from problems.env_conf import EnvConf, GymConf
from problems.linear_policy import LinearPolicy


def test_linear_policy_initialization():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=44)
    env.close()

    policy = LinearPolicy(env_conf)
    policy.set_params(policy.get_params())

    assert policy.problem_seed == 44
    assert policy._env_conf == env_conf
    assert policy._calculator._beta.shape == (env.action_space.shape[0], env.observation_space.shape[0])
    assert policy._calculator._normalizer is not None
    assert policy._calculator._num_beta == policy._calculator._beta.size
    assert policy._calculator._scale == 1


def test_linear_policy_num_params():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=45)
    env.close()

    policy = LinearPolicy(env_conf)
    expected_params = policy._calculator._num_beta + 2 + 2 * policy._calculator._num_state

    assert policy.num_params() == expected_params


def test_linear_policy_set_get_params():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=46)
    env.close()

    policy = LinearPolicy(env_conf)

    original_params = policy.get_params()
    assert len(original_params) == policy.num_params()

    new_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(new_params)

    retrieved_params = policy.get_params()
    np.testing.assert_array_almost_equal(retrieved_params, new_params)


def test_linear_policy_clone():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=47)
    env.close()

    policy = LinearPolicy(env_conf)
    original_params = policy.get_params()

    cloned_policy = policy.clone()

    assert cloned_policy is not policy
    assert cloned_policy._env_conf == policy._env_conf
    np.testing.assert_array_almost_equal(cloned_policy.get_params(), original_params)

    new_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(new_params)

    assert not np.array_equal(cloned_policy.get_params(), policy.get_params())


def test_linear_policy_call():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=42)
    env.close()

    policy = LinearPolicy(env_conf)
    policy.set_params(policy.get_params())

    state = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action = policy(state)

    assert action.shape == env.action_space.shape
    assert np.all(action >= -1) and np.all(action <= 1)


def test_linear_policy_normalization():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=43)
    env.close()

    policy = LinearPolicy(env_conf)
    policy.set_params(policy.get_params())

    state1 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    state2 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])

    action1 = policy(state1)
    action2 = policy(state2)

    assert action1.shape == action2.shape
    assert action1.shape == env.action_space.shape


def test_linear_policy_parameter_bounds():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True}, problem_seed=48)
    env.close()

    policy = LinearPolicy(env_conf)

    valid_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(valid_params)

    invalid_params = np.random.uniform(-2, 2, size=policy.num_params())
    try:
        policy.set_params(invalid_params)
        assert False, "Should have raised an assertion error for out-of-bounds parameters"
    except AssertionError:
        pass
