"""Benchmark functions (group 1f)."""

import numpy as np

# Re-export from 1f1 and 1f2
from .benchmark_functions_1f1 import (  # noqa: F401
    Langerman,
    Levy13,
)
from .benchmark_functions_1f2 import (  # noqa: F401
    RotatedHyperEllipsoid,
    SumOfDifferentPowers,
)
from .benchmark_util import mk_2d


class Bohachevsky1:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 100 * x
        x1, x2 = x[0], x[1]
        term1 = x1**2
        term2 = 2 * x2**2
        term3 = -0.3 * np.cos(3 * np.pi * x1)
        term4 = -0.4 * np.cos(4 * np.pi * x2)
        return term1 + term2 + term3 + term4 + 0.7


class ThreeHumpCamel:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 5
        x0 = x[0]
        x1 = x[1]
        return 2.0 * x0**2 - 1.05 * x0**4 + x0**6 / 6.0 + x0 * x1 + x1**2


class Trid:
    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = (d**2) * x
        term1 = np.sum((x - 1) ** 2)
        term2 = np.sum(x[1:] * x[:-1])
        return term1 - term2
