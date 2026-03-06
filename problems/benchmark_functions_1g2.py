"""Benchmark functions (group 1g2)."""

import numpy as np

from .benchmark_util import mk_4d


class PermDBeta:
    def __init__(self, beta=0.5):
        self.beta = beta

    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = d * x
        total = 0.0
        for i in range(1, d + 1):
            inner = 0.0
            for j in range(1, d + 1):
                term = (j**i + self.beta) * (((x[j - 1] / j) ** i) - 1)
                inner += term
            total += inner**2
        return total


class PowerSum:
    def __init__(self, b=None):
        self.b = np.array(b) if b is not None else np.array([8, 18, 44, 114])

    def __call__(self, x):
        x = np.asarray(x)
        x = mk_4d(x)
        d = len(x)
        x = (x + 1) * d / 2
        total = 0.0
        for i in range(1, d + 1):
            inner_sum = np.sum(x**i)
            total += (inner_sum - self.b[i - 1]) ** 2
        return total
