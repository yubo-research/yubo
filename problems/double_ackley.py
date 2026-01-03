import numpy as np

from problems.benchmark_functions_1 import Ackley


class DoubleAckley:
    def __init__(self):
        self._a0 = Ackley()
        self._a1 = Ackley()

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 1, x.shape
        assert len(x) % 2 == 0, len(x)
        h = len(x) // 2
        y0 = self._a0(x[:h])
        y1 = self._a1(x[h:])
        return np.asarray([y0, y1], dtype=np.float64)
