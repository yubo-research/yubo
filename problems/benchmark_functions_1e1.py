"""Benchmark functions (group 1e1)."""

import numpy as np

from .benchmark_util import mk_2d


class HolderTable:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 10
        x0 = x[0]
        x1 = x[1]
        part1 = np.sin(x0) * np.cos(x1)
        part2 = np.exp(np.abs(1 - np.sqrt(x0**2 + x1**2) / np.pi))
        return -np.abs(part1 * part2)


class Michalewicz:
    m = 10

    def __call__(self, x):
        x = np.pi * (1 + np.asarray(x)) / 2
        d = len(x)
        indices = np.arange(1, d + 1)
        sine_term1 = np.sin(x)
        sine_term2 = np.sin(indices * x**2 / np.pi) ** (2 * self.m)
        return -np.sum(sine_term1 * sine_term2)
