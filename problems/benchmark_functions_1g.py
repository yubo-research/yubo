"""Benchmark functions (group 1g)."""

import numpy as np

# Re-export from 1g1 and 1g2
from .benchmark_functions_1g1 import (  # noqa: F401
    Colville,
    DeJong5,
)
from .benchmark_functions_1g2 import (  # noqa: F401
    PermDBeta,
    PowerSum,
)
from .benchmark_util import mk_2d


class GoldsteinPrice:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 2 * x
        x1, x2 = x[0], x[1]
        part1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
        part2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
        return part1 * part2


class McCormick:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x1 = (4 + 1.5) / 2 * x[0] + (4 - 1.5) / 2
        x2 = (4 + 3) / 2 * x[1] + (4 - 3) / 2
        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        return term1 + term2 + term3 + term4 + 1


class Schwefel:
    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = 500 * x
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
