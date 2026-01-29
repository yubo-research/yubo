import numpy as np


class MockGymConf:
    def __init__(self, num_state, num_action):
        self.state_space = type("Space", (), {"shape": (num_state,)})()


class MockEnvConf:
    def __init__(self, num_state=4, num_action=2):
        self.problem_seed = 0
        self.gym_conf = MockGymConf(num_state, num_action)
        self.action_space = type("Space", (), {"shape": (num_action,)})()


def test_control_policy_factory():
    from problems.control_policy import ControlPolicyFactory

    factory = ControlPolicyFactory(use_layer_norm=True)
    env_conf = MockEnvConf()
    policy = factory(env_conf)
    assert policy is not None


def test_control_policy_init():
    from problems.control_policy import ControlPolicy

    env_conf = MockEnvConf()
    policy = ControlPolicy(env_conf)
    assert policy.num_params() > 0


def test_control_policy_forward():
    from problems.control_policy import ControlPolicy

    env_conf = MockEnvConf()
    policy = ControlPolicy(env_conf)
    state = np.random.rand(4)
    action = policy(state)
    assert action.shape == (2,)


def test_control_policy_get_set_params():
    from problems.control_policy import ControlPolicy

    env_conf = MockEnvConf()
    policy = ControlPolicy(env_conf)
    params = policy.get_params()
    policy.set_params(np.clip(params, -1, 1))


def test_control_policy_clone():
    from problems.control_policy import ControlPolicy

    env_conf = MockEnvConf()
    policy = ControlPolicy(env_conf)
    cloned = policy.clone()
    assert cloned is not policy
