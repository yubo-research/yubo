import numpy as np


class MockEnvConf:
    def __init__(self, num_dim):
        self.problem_seed = 0
        self.action_space = type("Space", (), {"low": np.zeros(num_dim)})()


def test_pure_function_policy_init():
    from problems.pure_function_policy import PureFunctionPolicy

    env_conf = MockEnvConf(5)
    policy = PureFunctionPolicy(env_conf)
    assert policy.num_params() == 5


def test_pure_function_policy_set_get_params():
    from problems.pure_function_policy import PureFunctionPolicy

    env_conf = MockEnvConf(5)
    policy = PureFunctionPolicy(env_conf)
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    policy.set_params(params)
    assert np.allclose(policy.get_params(), params)


def test_pure_function_policy_clone():
    from problems.pure_function_policy import PureFunctionPolicy

    env_conf = MockEnvConf(5)
    policy = PureFunctionPolicy(env_conf)
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    policy.set_params(params)
    cloned = policy.clone()
    assert np.allclose(cloned.get_params(), params)


def test_pure_function_policy_call():
    from problems.pure_function_policy import PureFunctionPolicy

    env_conf = MockEnvConf(5)
    policy = PureFunctionPolicy(env_conf)
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    policy.set_params(params)
    result = policy(0)
    assert np.allclose(result, params)
