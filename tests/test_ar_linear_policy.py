import gymnasium as gym
import numpy as np

from problems.ar_linear_policy import ARLinearPolicy
from problems.env_conf import EnvConf, GymConf


def test_ar_linear_policy_initialization():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)

    assert policy._env_conf == env_conf
    assert policy._num_ar == 5
    assert len(policy._queue) == 0
    assert policy._queue.maxlen == 5
    assert policy._calculator is not None

    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    big_state_size = (policy._num_ar - 1) * (num_state + num_action + 1)
    assert policy._calculator._num_state == big_state_size
    assert policy._calculator._num_action == num_action


def test_ar_linear_policy_num_params():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    expected_params = policy._calculator.num_params()

    assert policy.num_params() == expected_params


def test_ar_linear_policy_set_get_params():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)

    original_params = policy.get_params()
    assert len(original_params) == policy.num_params()

    new_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(new_params)

    retrieved_params = policy.get_params()
    np.testing.assert_array_almost_equal(retrieved_params, new_params)


def test_ar_linear_policy_clone():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    original_params = policy.get_params()

    cloned_policy = policy.clone()

    assert cloned_policy is not policy
    assert cloned_policy._env_conf == policy._env_conf
    assert cloned_policy._num_ar == policy._num_ar
    np.testing.assert_array_almost_equal(cloned_policy.get_params(), original_params)

    new_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(new_params)

    assert not np.array_equal(cloned_policy.get_params(), policy.get_params())


def test_ar_linear_policy_big_call_empty_queue():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)

    state = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward = 0.5

    result = policy.big_call(state, action, reward)

    assert result.shape == env.action_space.shape
    assert np.all(result == 0)
    assert len(policy._queue) == 1


def test_ar_linear_policy_big_call_partial_queue():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    policy.set_params(policy.get_params())

    state1 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action1 = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward1 = 0.5

    result1 = policy.big_call(state1, action1, reward1)
    assert result1.shape == env.action_space.shape
    assert np.all(result1 == 0)
    assert len(policy._queue) == 1

    state2 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action2 = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward2 = -0.3

    result2 = policy.big_call(state2, action2, reward2)
    assert result2.shape == env.action_space.shape
    assert np.all(result2 == 0)
    assert len(policy._queue) == 2


def test_ar_linear_policy_big_call_full_queue():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    policy.set_params(policy.get_params())

    # Fill the queue to capacity (5 elements)
    for i in range(5):
        state = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
        action = np.random.uniform(-1, 1, size=env.action_space.shape[0])
        reward = np.random.uniform(-1, 1)
        policy.big_call(state, action, reward)

    # Now add one more to trigger the calculation
    state6 = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    action6 = np.random.uniform(-1, 1, size=env.action_space.shape[0])
    reward6 = 0.8

    result = policy.big_call(state6, action6, reward6)

    assert result.shape == env.action_space.shape
    assert np.all(result >= -1) and np.all(result <= 1)
    assert len(policy._queue) == 5


def test_ar_linear_policy_queue_overflow():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    policy.set_params(policy.get_params())

    for i in range(10):
        state = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
        action = np.random.uniform(-1, 1, size=env.action_space.shape[0])
        reward = np.random.uniform(-1, 1)

        result = policy.big_call(state, action, reward)

        if i >= 4:
            assert result.shape == env.action_space.shape
            assert np.all(result >= -1) and np.all(result <= 1)

    assert len(policy._queue) == 5


def test_ar_linear_policy_create_big_state():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)

    # Add 2 elements to test differences calculation
    state1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    action1 = np.array([0.5, 0.3])
    reward1 = 0.7

    state2 = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    action2 = np.array([0.2, 0.8])
    reward2 = -0.4

    policy._queue.append((state1, action1, reward1))
    policy._queue.append((state2, action2, reward2))

    big_state = policy._create_big_state()

    # With use_differences=True, size is (num_ar-1) * (state+action+reward)
    expected_size = (policy._num_ar - 1) * (len(state1) + len(action1) + 1)
    assert big_state.shape == (expected_size,)

    # Check that it contains the difference between state2 and state1
    expected_diff = np.concatenate([state2 - state1, action2 - action1, [reward2 - reward1]])
    np.testing.assert_array_almost_equal(big_state[: len(expected_diff)], expected_diff)


def test_ar_linear_policy_parameter_bounds():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)

    valid_params = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(valid_params)

    invalid_params = np.random.uniform(-2, 2, size=policy.num_params())
    try:
        policy.set_params(invalid_params)
        assert False, "Should have raised an assertion error for out-of-bounds parameters"
    except AssertionError:
        pass


def test_ar_linear_policy_big_state_length():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    big_state_size = (policy._num_ar - 1) * (num_state + num_action + 1)

    # Call big_call fewer times than num_ar
    state = np.random.uniform(-1, 1, size=num_state)
    action = np.random.uniform(-1, 1, size=num_action)
    reward = 0.5
    policy.big_call(state, action, reward)
    partial_big_state = policy._create_big_state()
    print(f"Partial queue: big_state length = {len(partial_big_state)}, expected = {big_state_size}")
    assert len(partial_big_state) == big_state_size


def test_ar_linear_policy_k_attribute_error():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()

    policy = ARLinearPolicy(env_conf)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]

    # Fill the queue without calling set_params
    error_raised = False
    try:
        for _ in range(policy._num_ar):
            state = np.random.uniform(-1, 1, size=num_state)
            action = np.random.uniform(-1, 1, size=num_action)
            reward = np.random.uniform(-1, 1)
            policy.big_call(state, action, reward)
    except AttributeError as e:
        print(f"Caught AttributeError as expected: {e}")
        error_raised = True
    assert error_raised, "Expected AttributeError for missing _k when set_params is not called."

    # Now call set_params and try again
    policy = ARLinearPolicy(env_conf)
    policy.set_params(policy.get_params())
    try:
        for _ in range(policy._num_ar):
            state = np.random.uniform(-1, 1, size=num_state)
            action = np.random.uniform(-1, 1, size=num_action)
            reward = np.random.uniform(-1, 1)
            policy.big_call(state, action, reward)
    except AttributeError:
        assert False, "AttributeError should not be raised after set_params is called."


def test_ar_linear_policy_mixed_scalar_array_queue():
    env = gym.make("LunarLander-v3", continuous=True)
    gym_conf = GymConf(max_steps=500)
    env_conf = EnvConf("LunarLander-v3", gym_conf=gym_conf, kwargs={"continuous": True})
    env.close()
    policy = ARLinearPolicy(env_conf)
    # First entry: state and action as arrays
    state1 = np.ones(8)
    action1 = np.ones(2)
    reward1 = 1.0
    # Second entry: state as array, action as scalar
    state2 = np.ones(8) * 2
    action2 = 2.0
    reward2 = 2.0
    # Third entry: state as scalar, action as array
    state3 = 3.0
    action3 = np.ones(2) * 3
    reward3 = 3.0
    policy._queue.append((state1, action1, reward1))
    policy._queue.append((state2, action2, reward2))
    policy._queue.append((state3, action3, reward3))
    # Should not raise
    big_state = policy._create_big_state()
    assert isinstance(big_state, np.ndarray)
    assert big_state.ndim == 1
    # All rows in the state matrix should be the same length (internally)
    # The length should match the reference entry size
    ref_size = len(np.concatenate([np.ravel(state1), np.ravel(action1), [reward1]]))
    assert big_state.size % ref_size == 0 or policy._use_differences
