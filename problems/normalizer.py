import numpy as np


class Normalizer:
    def __init__(self, shape, num_init=1, init_mean=0.0, init_var=1.0):
        if isinstance(init_mean, np.ndarray):
            assert init_var.min() > 0, init_var.min()
        self._x = init_mean + np.zeros(shape=shape)
        self._x2 = init_var * np.ones(shape=shape)
        self._num = num_init

    def update(self, x):
        self._x += x
        self._x2 += x * x
        self._num += 1

    def mean_and_std(self):
        m = self._x / self._num
        v = self._x2 / self._num - m**2
        return m, np.sqrt(np.maximum(0.0, v))
