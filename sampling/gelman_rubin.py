import numpy as np


class GelmanRubin:
    def __init__(self):
        self._x = []

    def append(self, x):
        self._x.append(x)

    def get(self):
        x = np.array(self._x)

        N, J, D = x.shape
        x_bar = x.mean(axis=1)
        x_star = x_bar.mean(axis=0, keepdims=True)
        B = N / (J - 1) * ((x_bar - x_star) ** 2).sum(axis=0)
        W = 1 / J / (N - 1) * ((x - np.expand_dims(x_bar, 1)) ** 2).sum(axis=0).sum(axis=0)

        B = np.prod(B) ** (1 / D)
        W = np.prod(W) ** (1 / D)

        num = (N - 1) / N * W + B / N
        return num / W
