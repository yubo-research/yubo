import numpy as np


class Normalizer:
    def __init__(self, shape):
        self._x = np.zeros(shape=shape)
        self._x2 = np.zeros(shape=shape)
        self._num = 0

    def update(self, x):
        self._x += x
        self._x2 += x * x
        self._num += 1

    def mean_and_std(self):
        m = self._x / self._num
        v = self._x2 / self._num - m**2
        return m, np.sqrt(np.maximum(0.0, v))
