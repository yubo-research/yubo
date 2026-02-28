"""Benchmark functions (group 1e2)."""

import numpy as np

from .benchmark_util import mk_4d


class Powell:
    def __call__(self, x):
        x = 0.5 + 4.5 * np.asarray(x)
        d = len(x)
        if d % 4 != 0:
            x = mk_4d(x)
            d = len(x)
        x_grouped = x.reshape(-1, 4)
        term1 = (x_grouped[:, 0] + 10 * x_grouped[:, 1]) ** 2
        term2 = 5 * (x_grouped[:, 2] - x_grouped[:, 3]) ** 2
        term3 = (x_grouped[:, 1] - 2 * x_grouped[:, 2]) ** 4
        term4 = 10 * (x_grouped[:, 0] - x_grouped[:, 3]) ** 4
        return np.sum(term1 + term2 + term3 + term4)


class Rastrigin:
    def __call__(self, x):
        x = x * 5.12
        num_dim = len(x)
        return 10 + np.sum(x**2 - 10 * np.cos(np.pi * 2 * x)) / num_dim
