import numpy as np


class GelmanRubin:
    def __init__(self, burn_in=10):
        self._burn_in = burn_in
        self._x = []

    def append(self, x):
        self._x.append(x)

    def r_hat(self):
        N = len(self._x)
        if N < self._burn_in + 3:
            return np.inf

        x = np.array(self._x[self._burn_in :])
        N, J, D = x.shape

        x_bar = x.mean(axis=1)
        x_star = x_bar.mean(axis=0, keepdims=True)
        B = N / (J - 1) * ((x_bar - x_star) ** 2).sum(axis=0)
        W = (
            1
            / J
            / (N - 1)
            * ((x - np.expand_dims(x_bar, 1)) ** 2).sum(axis=0).sum(axis=0)
        )

        B = np.prod(B) ** (1 / D)
        W = np.prod(W) ** (1 / D)

        num = (N - 1) / N * W + B / N
        return num / W
