import numpy as np


class Sphere:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        return -(((x - self._x_0) ** 2)).mean()
