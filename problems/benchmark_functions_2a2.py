"""Benchmark functions (group 2a2)."""

import numpy as np


class Salomon:
    def __call__(self, x):
        x = 100 * np.asarray(x)
        r = np.linalg.norm(x)
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r


class Sum_Squares:
    def __call__(self, x):
        x = 10 * np.asarray(x)
        return np.sum((x**2) * np.arange(1, len(x) + 1))
