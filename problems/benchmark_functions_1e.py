"""Benchmark functions (group 1e)."""

import numpy as np

# Re-export from 1e1 and 1e2
from .benchmark_functions_1e1 import (  # noqa: F401
    HolderTable,
    Michalewicz,
)
from .benchmark_functions_1e2 import (  # noqa: F401
    Powell,
    Rastrigin,
)
from .benchmark_util import mk_2d


class GrLee12:
    def __call__(self, x):
        x = mk_2d(x)
        x = x[0] + 1.5
        return np.sin(10.0 * np.pi * x) / (2.0 * x) + (x - 1.0) ** 4


class Rosenbrock:
    def __call__(self, x):
        x = x * 2.048
        part = 0
        num_dim = len(x)
        for i in range(num_dim - 1):
            part += (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return 100.0 * part / num_dim


class Shubert:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 5.12
        x0 = x[0]
        x1 = x[1]
        part1 = 0
        part2 = 0
        for i in range(1, 6):
            new1 = i * np.cos((i + 1) * x0 + i)
            new2 = i * np.cos((i + 1) * x1 + i)
            part1 += new1
            part2 += new2
        return part1 * part2
