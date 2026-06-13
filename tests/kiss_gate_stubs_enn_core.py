import numpy as np


class _Posterior:
    def __init__(self, n):
        self.mu = np.zeros((n, 1))
        self.se = np.ones((n, 1))


class _ENN:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self._n = 0

    def add(self, x, y, yvar=None):
        _ = y, yvar
        self._n += len(x)

    def __len__(self):
        return self._n

    def posterior(self, x, params=None, flags=None):
        _ = params, flags
        return _Posterior(len(x))

    def train_rows_at(self, idx):
        n = len(idx)
        return np.zeros((n, 1)), np.zeros(n), np.ones(n)
