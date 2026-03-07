"""Benchmark functions (group 1b)."""

import numpy as np

# Re-export from 1f
from .benchmark_functions_1f import (  # noqa: F401
    Bohachevsky1,
    Langerman,
    Levy13,
    RotatedHyperEllipsoid,
    SumOfDifferentPowers,
    ThreeHumpCamel,
    Trid,
)
from .benchmark_util import mk_2d, mk_4d


class Shekel:
    def __init__(self):
        self.beta = 0.1 * np.array((1, 2, 2, 4, 4, 6, 3, 7, 5, 5)).T
        self.C = np.array(
            (
                4,
                1,
                8,
                6,
                3,
                2,
                5,
                8,
                6,
                7,
                4,
                1,
                8,
                6,
                7,
                9,
                3,
                1,
                2,
                3.6,
                4,
                1,
                8,
                6,
                3,
                2,
                5,
                8,
                6,
                7,
                4,
                1,
                8,
                6,
                7,
                9,
                3,
                1,
                2,
                3.6,
            )
        ).reshape(4, 10)

    def __call__(self, x):
        if len(x) != 4:
            x = mk_4d(x)
        x = x * 5 + 5
        m = self.C.shape[1]
        outer = 0
        for i in range(m):
            bi = self.beta[i]
            inner = 0
            for j in range(4):
                inner += (x[j] - self.C[j][i]) ** 2
            outer += 1 / (inner + bi)
        return outer


class SixHumpCamel:
    def __call__(self, x):
        x = mk_2d(x)
        x0 = x[0] * 3
        x1 = x[1] * 2
        return (4 - 2.1 * x0**2 + x0**4 / 3) * x0**2 + x0 * x1 + (4 * x1**2 - 4) * x1**2


class StybTang:
    def __call__(self, x):
        x = x * 5
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
