"""Benchmark functions (group 1f1)."""

import numpy as np

from .benchmark_util import mk_2d


class Langerman:
    def __init__(self):
        self.c = np.array([1, 2, 5, 2, 3])
        self.A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        self.m = 5

    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 5 * x + 10
        outer = 0.0
        for i in range(self.m):
            inner = np.sum((x - self.A[i, :]) ** 2)
            new = self.c[i] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
            outer += new
        return outer


class Levy13:
    def __call__(self, x):
        x = np.asarray(x)
        x = 10 * x
        x = mk_2d(x)
        x1 = x[0]
        x2 = x[1]
        term1 = np.sin(3 * np.pi * x1) ** 2
        term2 = (x1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x2) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x2) ** 2)
        return term1 + term2 + term3
