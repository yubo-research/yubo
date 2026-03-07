"""Benchmark functions (group 1f2)."""

import numpy as np


class RotatedHyperEllipsoid:
    def __call__(self, x):
        x = np.asarray(x)
        x = 65.536 * x
        return np.sum(np.cumsum(x**2))


class SumOfDifferentPowers:
    def __call__(self, x):
        x = np.asarray(x)
        return np.sum(np.abs(x) ** (np.arange(1, len(x) + 1) + 1))
