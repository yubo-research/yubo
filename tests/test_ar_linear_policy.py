import gymnasium as gym
import numpy as np

from problems.ar_linear_policy import ARLinearPolicy
from problems.env_conf import EnvConf, GymConf


def test_ar_linear_policy_initialization():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf, num_ar=3, use_differences=True)

    assert policy._num_ar == 3
    assert policy._use_differences == True
    assert policy._queue.maxlen == 3
    assert len(policy._queue) == 0

    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    big_state_size = (policy._num_ar - 1) * (num_state + num_action + 1)
    assert policy._calculator._num_state == big_state_size


def test_ar_linear_policy_parameter_management():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf, num_ar=2, use_differences=False)

    original_params = policy.get_params()
    assert len(original_params) == policy.num_params()

    new_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(new_params)

    retrieved_params = policy.get_params()
    np.testing.assert_array_almost_equal(retrieved_params, new_params)


def test_ar_linear_policy_queue_behavior():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf, num_ar=2, use_differences=False)
    policy.set_params(policy.get_params())

    state1 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action1 = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward1 = 0.5

    result1 = policy.big_call(state1, action1, reward1)
    assert result1.shape == env.action_space.shape
    assert len(policy._queue) == 1

    state2 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action2 = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward2 = -0.3

    result2 = policy.big_call(state2, action2, reward2)
    assert result2.shape == env.action_space.shape
    assert len(policy._queue) == 2

    # Test queue overflow
    state3 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action3 = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward3 = 0.8

    result3 = policy.big_call(state3, action3, reward3)
    assert result3.shape == env.action_space.shape
    assert len(policy._queue) == 2  # Queue maxlen is 2


def test_ar_linear_policy_differences_mode():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf, num_ar=3, use_differences=True)
    policy.set_params(policy.get_params())

    # Add first entry - should return actual calculations (padded with zeros)
    state1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    action1 = np.array([0.5, 0.3])
    reward1 = 0.7

    result1 = policy.big_call(state1, action1, reward1)
    assert result1.shape == env.action_space.shape
    assert np.all(result1 >= -1) and np.all(result1 <= 1)

    # Add second entry - should return actual calculations (padded with zeros)
    state2 = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    action2 = np.array([0.2, 0.8])
    reward2 = -0.4

    result2 = policy.big_call(state2, action2, reward2)
    assert result2.shape == env.action_space.shape
    assert np.all(result2 >= -1) and np.all(result2 <= 1)

    # Add third entry - should now calculate differences
    state3 = np.array([17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    action3 = np.array([0.9, 0.1])
    reward3 = 0.2

    result3 = policy.big_call(state3, action3, reward3)
    assert result3.shape == env.action_space.shape
    assert np.all(result3 >= -1) and np.all(result3 <= 1)
    assert len(policy._queue) == 3


def test_ar_linear_policy_mixed_input_types():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf, num_ar=2, use_differences=False)
    policy.set_params(policy.get_params())

    # Test with mixed scalar and array inputs
    state1 = np.ones(8)
    action1 = np.ones(2)
    reward1 = 1.0

    state2 = np.ones(8) * 2
    action2 = 2.0  # Scalar action
    reward2 = 2.0

    result1 = policy.big_call(state1, action1, reward1)
    result2 = policy.big_call(state2, action2, reward2)

    assert result1.shape == env.action_space.shape
    assert result2.shape == env.action_space.shape
    assert np.all(result1 >= -1) and np.all(result1 <= 1)
    assert np.all(result2 >= -1) and np.all(result2 <= 1)
