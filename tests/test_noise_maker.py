import numpy as np


class MockEnv:
    def __init__(self):
        self.observation_space = type("Space", (), {"low": np.array([0.0])})()
        self.action_space = type("Space", (), {"shape": (2,)})()
        self._step_count = 0

    def step(self, action):
        self._step_count += 1
        return 0, 1.0, False, None

    def reset(self, seed):
        return 0, None

    def close(self):
        pass


def test_noise_maker_observation_space():
    from problems.noise_maker import NoiseMaker

    env = MockEnv()
    nm = NoiseMaker(env, 0.1, num_measurements=10)
    assert nm.observation_space is not None


def test_noise_maker_action_space():
    from problems.noise_maker import NoiseMaker

    env = MockEnv()
    nm = NoiseMaker(env, 0.1, num_measurements=10)
    assert nm.action_space is not None


def test_noise_maker_step():
    from problems.noise_maker import NoiseMaker

    env = MockEnv()
    nm = NoiseMaker(env, 0.1, num_measurements=10)
    nm.reset(seed=42)
    result = nm.step(np.array([0.5, 0.5]))
    assert len(result) == 4
