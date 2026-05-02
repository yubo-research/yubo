import numpy as np


class _Space:
    shape = (1,)


class _Env:
    observation_space = _Space()
    action_space = _Space()

    def __init__(self):
        self.n = 0

    def reset(self, seed=None):
        return np.zeros(1), {"seed": seed}

    def step(self, action):
        self.n += 1
        return np.zeros(1), 1.0, False, self.n > 1, {}

    def close(self):
        return None
