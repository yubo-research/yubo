import numpy as np


class RandomDesigner:
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, data):
        policy = self._policy.clone()
        p = policy.get_params()
        p = np.random.uniform(-1, 1, size=p.shape)
        policy.set_params(p)
        return policy
