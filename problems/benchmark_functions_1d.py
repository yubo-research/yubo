"""Additional benchmark functions for optimization (group 1d)."""

import numpy as np

# Re-export from 1d1 and 1d2
from .benchmark_functions_1d1 import (  # noqa: F401
    Branin,
    Bukin,
)
from .benchmark_functions_1d2 import (  # noqa: F401
    CrossInTray,
    DropWave,
)
from .benchmark_util import mk_2d


class Beale:
    def __call__(self, x):
        x = mk_2d(x)
        x = 4.5 * x
        part1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        part2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        part3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return part1 + part2 + part3


class DixonPrice:
    def __call__(self, x):
        x = x * 10
        part1 = (x[0] - 1) ** 2
        sum_terms = 0
        for i in range(2, len(x) + 1):
            xnew = x[i - 1]
            xold = x[i - 2]
            new = i * (2 * xnew**2 - xold) ** 2
            sum_terms += new
        return part1 + sum_terms


class EggHolder:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 511
        x1 = x[0]
        x2 = x[1]
        part1 = -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47.0))))
        return part1 + part2
