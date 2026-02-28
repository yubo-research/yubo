"""Benchmark functions (group 2)."""

import numpy as np

from .benchmark_functions_2a import (  # noqa: F401
    Brown,
    Himmelblau,
    Matyas,
    Salomon,
    Sum_Squares,
    Whitley,
    Zettl,
)
from .benchmark_functions_3 import Zakharov  # noqa: F401
from .benchmark_util import mk_2d

# Requirements:
# - x in [-1,1]**num_dim
# - have *minima* [not maxima]
# - support any number of dimensions as input


class Alpine:
    """
    See: https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/162-alpine-function-no-1
    """

    def __call__(self, x):
        x = 5 * (1 + np.asarray(x))
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


class Easom:
    """
    See: https://www.sfu.ca/~ssurjano/easom.html
    """

    def __call__(self, x):
        x = 100 * mk_2d(x)
        y = x[1]
        x = x[0]
        return np.sum(-np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2) - (y - np.pi) ** 2))


class Booth:
    def __call__(self, x):
        x = 10 * mk_2d(x)
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


# Zakharov moved to benchmark_functions_3.py
# Other functions moved to benchmark_functions_2a.py
