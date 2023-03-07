import numpy as np


class Rastrigin:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        Z= 10*x.shape[-1]+np.sum(x[::-1]**2-10*np.cos(2*np.pi*x[::-1]))
        return Z.mean()
