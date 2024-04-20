import numpy as np


class GelmanRubin:
    def __init__(self):
        self._x = []

    def append(self, x):
        self._x.append(x)

    def get(self):
        x = np.array(self._x)
        N, J = x.shape
        x_bar = x.mean(axis=1)
        x_star = x_bar.mean()
        B = N / (J - 1) * ((x_bar - x_star) ** 2).sum()
        W = 1 / J / (N - 1) * ((x - x_bar[:, None]) ** 2).sum()
        num = (N - 1) / N * W + B / N
        return num / W
