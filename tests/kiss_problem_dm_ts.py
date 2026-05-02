import numpy as np


class _TS:
    def __init__(self, last=False):
        self.observation = {"state": np.zeros((4,), dtype=np.float32)}
        self.reward = 1.0
        self._last = last
        self.discount = 1.0

    def last(self):
        return self._last


class _Spec:
    def __init__(self, shape, minimum=None, maximum=None, dtype=np.float32):
        self.shape = shape
        self.minimum = minimum
        self.maximum = maximum
        self.dtype = dtype
