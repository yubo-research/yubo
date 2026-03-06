"""Benchmark functions (group 1c)."""

import numpy as np

# Re-export from 1g
from .benchmark_functions_1g import (  # noqa: F401
    Colville,
    DeJong5,
    GoldsteinPrice,
    McCormick,
    PermDBeta,
    PowerSum,
    Schwefel,
)
from .benchmark_util import mk_2d


class Perm:
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
                term = (j + self.beta) * (x[j - 1] ** i - 1 / (j**i))
                inner += term
            total += inner**2
        return total


class Schaffer2:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 100 * x
        x1, x2 = x[0], x[1]
        numerator = np.sin(x1**2 - x2**2) ** 2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        return 0.5 + numerator / denominator


class Schaffer4:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 100 * x
        x1, x2 = x[0], x[1]
        diff = np.abs(x1**2 - x2**2)
        numerator = np.cos(np.sin(diff)) ** 2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        return 0.5 + numerator / denominator
