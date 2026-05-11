import numpy as np


class _Posterior:
    def __init__(self, n):
        self.mu = np.zeros((n, 1))
        self.se = np.ones((n, 1))


class _ENN:
    def posterior(self, x, params=None, flags=None):
        _ = params, flags
        return _Posterior(len(x))
