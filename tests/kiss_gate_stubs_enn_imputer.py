import numpy as np


class _PosteriorImputer:
    def __init__(self, n):
        self.mu = np.zeros((n, 1))
        self.se = np.zeros((n, 1))


class _ENNImputer:
    def posterior(self, x, params=None, flags=None):
        _ = params, flags
        return _PosteriorImputer(len(x))
