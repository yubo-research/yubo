"""Benchmark functions (group 2a1)."""

from .benchmark_util import mk_2d


class Himmelblau:
    def __call__(self, x):
        x = 6 * mk_2d(x)
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class Matyas:
    def __call__(self, x):
        x = 10 * mk_2d(x)
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
