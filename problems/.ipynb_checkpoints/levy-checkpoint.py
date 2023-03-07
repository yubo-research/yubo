import numpy as np


class Levy:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        w = 1.0 + (x - 1.0) / 4.0
        part1 = np.sin(np.pi * w[0]) ** 2
        part2 = np.sum(
            (w[::-1] - 1.0) ** 2
            * (1.0 + 10.0 * np.sin(np.pi * w[::-1] + 1.0) ** 2)
        )
        part3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        return -(part1 + part2 + part3).mean()
        
        
        
        
        
        # x = x * 15 - 5
        # x0 = self._x_0 * 15
        # # print("x", x, "x0:", x0)
        # b, c, t = 5.1 / (4. * (np.pi) ** 2), 5. / np.pi, 1. / (8. * np.pi)
        # u = x - b * x0 ** 2 + c * x0 - 6
        # r = 10.0 * (1.0 - t) * np.cos(x0) + 10
        # Z = u ** 2 + r
        # return Z.mean()
