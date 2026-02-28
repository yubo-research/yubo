"""Benchmark functions (group 1a)."""

import numpy as np

# Re-export from 1e
from .benchmark_functions_1e import (  # noqa: F401
    GrLee12,
    HolderTable,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shubert,
)
from .benchmark_util import mk_4d


class Griewank:
    def __call__(self, x):
        x = x * 600
        part1 = np.sum(x**2 / 4000.0)
        num_dim = len(x)
        part2 = np.prod(np.cos(x / np.sqrt((1 + np.arange(num_dim)))))
        return part1 - part2 + 1


class Hartmann:
    def __init__(self):
        self.A = {
            3: np.array((3, 10, 30, 0.1, 10, 35, 3, 10, 30, 0.1, 10, 35)).reshape(4, 3),
            4: np.array(
                (
                    10,
                    3,
                    17,
                    3.5,
                    1.7,
                    8,
                    0.05,
                    10,
                    17,
                    0.1,
                    8,
                    14,
                    3,
                    3.5,
                    1.7,
                    10,
                    17,
                    8,
                    17,
                    8,
                    0.05,
                    10,
                    0.1,
                    14,
                )
            ).reshape(4, 6),
        }
        self.P = {
            3: 10 ** (-4) * np.array((3689, 1170, 2673, 4699, 4387, 7470, 1091, 8732, 5547, 381, 5743, 8828)).reshape(4, 3),
            4: 10 ** (-4)
            * np.array(
                (
                    1312,
                    1696,
                    5569,
                    124,
                    8283,
                    5886,
                    2329,
                    4135,
                    8307,
                    3736,
                    1004,
                    9991,
                    2348,
                    1451,
                    3522,
                    2883,
                    3047,
                    6650,
                    4047,
                    8828,
                    8732,
                    5743,
                    1091,
                    381,
                )
            ).reshape(4, 6),
        }
        self.ALPHA = [1.0, 1.2, 3.0, 3.2]

    def __call__(self, x):
        num_dim = len(x)
        if num_dim not in self.A:
            if len(x) == 1:
                x = np.array([x[0]] * 3)
            else:
                x = mk_4d(x)
            num_dim = len(x)

        A = self.A[num_dim]
        P = self.P[num_dim]

        x = 0.5 * x + 0.5
        outer = 0
        for i in range(4):
            inner = 0
            for j in range(num_dim):
                inner += A[i][j] * (x[j] - P[i][j]) ** 2
            new = self.ALPHA[i] * np.exp(-inner)
            outer += new
        if num_dim == 3:
            return -outer
        if num_dim == 4:
            return (1.1 - outer) / 0.839
        if num_dim == 6:
            return -(2.58 + outer) / 1.94


class Levy:
    def __call__(self, x):
        x = 10 * x
        w = 1.0 + (x - 1.0) / 4.0
        part1 = np.sin(np.pi * w[0]) ** 2
        part2 = 0
        num_dim = len(x)
        for i in range(num_dim - 1):
            part2 += (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2)
        part3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        return part1 + part2 + part3
