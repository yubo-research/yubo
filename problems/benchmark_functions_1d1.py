"""Additional benchmark functions (group 1d1)."""

import numpy as np

from .benchmark_util import mk_2d


class Branin:
    def __init__(self):
        self.a = 1
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

    def __call__(self, x):
        x = mk_2d(x)
        x1 = 7.5 * x[0] + 2.5
        x2 = 7.5 * x[1] + 7.5
        return self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2 + self.s * (1 - self.t) * np.cos(x1) + self.s


class Bukin:
    def __call__(self, x):
        x = mk_2d(x)
        x0 = x[0] * 5 - 10
        x1 = x[1] * 3
        return 100.0 * np.sqrt(np.abs(x1 - 0.01 * x0**2)) + 0.01 * np.abs(x0 + 10.0)
