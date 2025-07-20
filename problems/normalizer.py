import numpy as np


class Normalizer:
    def __init__(self, shape, num_init=1, init_mean=0.0, init_var=1.0):
        if isinstance(init_mean, np.ndarray) and isinstance(init_var, np.ndarray):
            assert init_var.min() > 0, init_var.min()
        assert isinstance(num_init, int), num_init
        assert num_init >= 0, num_init
        if num_init == 0:
            self._x = np.zeros(shape=shape)
            self._x2 = np.zeros(shape=shape)
            self._num = 0
        else:
            self._x = init_mean * num_init + np.zeros(shape=shape)
            self._x2 = (init_var + init_mean**2) * num_init
            assert self._x2.min() >= 0, self._x2.min()
            self._num = num_init

    def update(self, x):
        self._x += x
        self._x2 += x * x
        self._num += 1

    def mean_and_std(self):
        m = self._x / self._num
        v = self._x2 / self._num - m**2
        assert v.min() >= -1e-6, v.min()

        # print("LS:", m.mean(), np.sqrt(v.mean()))
        return m, np.sqrt(np.maximum(0.0, v))
