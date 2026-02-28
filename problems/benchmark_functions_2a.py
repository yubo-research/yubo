"""Benchmark functions (group 2a)."""

import numpy as np

# Re-export from 2a1 and 2a2
from .benchmark_functions_2a1 import (  # noqa: F401
    Himmelblau,
    Matyas,
)
from .benchmark_functions_2a2 import (  # noqa: F401
    Salomon,
    Sum_Squares,
)
from .benchmark_util import mk_2d


class Brown:
    """
    See: https://www.indusmic.com/post/brown-function
    """

    def __call__(self, x):
        x = 1.5 + 2.5 * np.asarray(x)
        d = len(x)
        if d < 2:
            x = mk_2d(x)
            d = 2

        xi_squared = x[:-1] ** 2
        xi_plus_1_squared = x[1:] ** 2

        term1 = xi_squared * (xi_plus_1_squared + 1)
        term2 = xi_plus_1_squared * (xi_squared + 1)

        return np.sum(term1 + term2)


class Whitley:
    """
    See: http://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Whitley
    """

    def __call__(self, x):
        x = 10.24 * np.asarray(x)
        d = len(x)

        i_indices, j_indices = np.meshgrid(range(d), range(d), indexing="ij")
        xi_squared = x[i_indices] ** 2
        xj = x[j_indices]

        term1 = (100 * (xi_squared - xj) ** 2 + (1 - xj) ** 2) ** 2 / 4000
        term2 = -np.cos(100 * (xi_squared - xj) ** 2 + (1 - xj) ** 2)

        return np.sum(term1 + term2 + 1)


class Zettl:
    def __call__(self, x):
        x = 2 + 3 * mk_2d(x)
        return (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]
