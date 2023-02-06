import numpy as np


class Ackley:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        return -(20* np.exp(-0.2 * np.sqrt(0.5 * (x**2 +self._x_0**2)))-np.exp(0.5 * (np.cos(2* np.pi * self._x_0)))+np.e +20).mean()
