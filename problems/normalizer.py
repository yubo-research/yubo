import numpy as np


class Normalizer:
    def __init__(self, shape, decay=0.99):
        if decay is None:
            self._decay = None
        else:
            self._decay = float(decay)
            assert 0 <= self._decay < 1, self._decay
        self._x = np.zeros(shape=shape)
        self._x2 = np.zeros(shape=shape)
        self._num = 0.0

    def update(self, x):
        if self._decay is None:
            self._x += x
            self._x2 += x * x
            self._num += 1.0
            return

        self._x = self._decay * self._x + x
        self._x2 = self._decay * self._x2 + x * x
        self._num = self._decay * self._num + 1.0

    def mean_and_std(self):
        assert self._num > 0, self._num
        m = self._x / self._num
        v = self._x2 / self._num - m**2
        return m, np.sqrt(np.maximum(0.0, v))

    def clone(self):
        n = Normalizer(shape=self._x.shape, decay=self._decay)
        n._x = self._x.copy()
        n._x2 = self._x2.copy()
        n._num = self._num
        return n
