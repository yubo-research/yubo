"""Benchmark functions (group 1g1)."""

import numpy as np

from .benchmark_util import mk_2d, mk_4d


class Colville:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_4d(x)
        x = 10 * x
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        term1 = 100 * (x1**2 - x2) ** 2
        term2 = (x1 - 1) ** 2
        term3 = 90 * (x3**2 - x4) ** 2
        term4 = (x3 - 1) ** 2
        term5 = 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        term6 = 19.8 * (x2 - 1) * (x4 - 1)
        return term1 + term2 + term3 + term4 + term5 + term6


class DeJong5:
    def __init__(self):
        a_vals = np.array([-32, -16, 0, 16, 32])
        self.a = np.array([[x, y] for x in a_vals for y in a_vals]).T

    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 65.536 * x
        x1, x2 = x[0], x[1]
        sum_terms = 0.0
        for i in range(25):
            ai1 = self.a[0, i]
            ai2 = self.a[1, i]
            denom = i + 1 + (x1 - ai1) ** 6 + (x2 - ai2) ** 6
            sum_terms += 1 / denom
        return 1.0 / (0.002 + sum_terms)
