import numpy as np


class _DummySpec:
    def __init__(self, *, shape, dtype=np.float32, minimum=-1.0, maximum=1.0):
        self.shape = shape
        self.dtype = dtype
        self.minimum = np.full(shape, minimum, dtype=np.float32)
        self.maximum = np.full(shape, maximum, dtype=np.float32)


class _DummyTimeStep:
    def __init__(self, observation, reward, discount, is_last):
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._is_last = is_last

    def last(self):
        return self._is_last


class _DummyGlobal:
    offwidth = 640
    offheight = 480
