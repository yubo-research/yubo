import numpy as np


class Branin:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        x = x * 15 - 5
        x0 = self._x_0 * 15
        # print("x", x, "x0:", x0)
        b, c, t = 5.1 / (4. * (np.pi) ** 2), 5. / np.pi, 1. / (8. * np.pi)
        u = x - b * x0 ** 2 + c * x0 - 6
        r = 10.0 * (1.0 - t) * np.cos(x0) + 10
        Z = u ** 2 + r
        return Z.mean()
