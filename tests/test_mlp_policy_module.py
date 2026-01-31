import numpy as np


class MockGymConf:
    def __init__(self, num_state, num_action):
        self.state_space = type("Space", (), {"shape": (num_state,)})()


class MockEnvConf:
    def __init__(self, num_state=4, num_action=2):
        self.problem_seed = 0
        self.gym_conf = MockGymConf(num_state, num_action)
        self.action_space = type("Space", (), {"shape": (num_action,)})()


def test_mlp_policy_factory():
    from problems.mlp_policy import MLPPolicyFactory

    factory = MLPPolicyFactory((16, 8))
    env_conf = MockEnvConf()
    policy = factory(env_conf)
    assert policy is not None


def test_mlp_policy_factory_with_rnn():
    from problems.mlp_policy import MLPPolicyFactory

    factory = MLPPolicyFactory((), rnn_hidden_size=4)
    env_conf = MockEnvConf()
    policy = factory(env_conf)
    assert policy is not None


def test_mlp_policy_forward():
    from problems.mlp_policy import MLPPolicy

    env_conf = MockEnvConf()
    policy = MLPPolicy(env_conf, (16, 8))
    state = np.random.rand(4)
    action = policy(state)
    assert action.shape == (2,)


def test_mlp_policy_num_params():
    from problems.mlp_policy import MLPPolicy

    env_conf = MockEnvConf()
    policy = MLPPolicy(env_conf, (16, 8))
    assert policy.num_params() > 0


def test_mlp_policy_clone():
    from problems.mlp_policy import MLPPolicy

    env_conf = MockEnvConf()
    policy = MLPPolicy(env_conf, (16, 8))
    cloned = policy.clone()
    assert cloned is not policy
