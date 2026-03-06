"""Additional benchmark functions (group 1d2)."""

import numpy as np

from .benchmark_util import mk_2d


class CrossInTray:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 9 + 1
        x0 = x[0]
        x1 = x[1]
        part1 = np.abs(np.sin(x0) * np.sin(x1) * np.exp(np.abs(100.0 - np.sqrt(x0**2 + x1**2) / np.pi))) + 1.0
        part2 = np.power(part1, 0.1)
        return -0.0001 * part2


class DropWave:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 5.12
        x0 = x[0]
        x1 = x[1]
        sum2 = x0**2 + x1**2
        part1 = 1.0 + np.cos(12.0 * np.sqrt(sum2))
        part2 = 0.5 * sum2 + 2.0
        return -part1 / part2
