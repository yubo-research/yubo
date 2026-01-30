import numpy as np

# Re-export classes from split files
from .benchmark_functions_1a import (  # noqa: F401
    Griewank,
    GrLee12,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shubert,
)
from .benchmark_functions_1b import (  # noqa: F401
    Bohachevsky1,
    Langerman,
    Levy13,
    RotatedHyperEllipsoid,
    Shekel,
    SixHumpCamel,
    StybTang,
    SumOfDifferentPowers,
    ThreeHumpCamel,
    Trid,
)
from .benchmark_functions_1c import (  # noqa: F401
    Colville,
    DeJong5,
    GoldsteinPrice,
    McCormick,
    Perm,
    PermDBeta,
    PowerSum,
    Schaffer2,
    Schaffer4,
    Schwefel,
)

# Requirements:
# - x in [-1,1]**num_dim
# - have *minima* [not maxima]
# - support any number of dimensions as input
from .benchmark_util import mk_2d


class Sphere3:
    def __call__(self, x):
        return ((x - 0.3) ** 2).mean()


"""
See: https://www.sfu.ca/~ssurjano/
"""


class Sphere:
    def __call__(self, x):
        return (x**2).mean()


class Ackley:
    def __init__(self):
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * np.pi

    def __call__(self, x):
        x = 32.768 * x
        return (
            -self.a * np.exp(-self.b * np.sqrt((x**2).mean()))
            - np.exp(np.cos(self.c * x).mean())
            + self.a
            + np.e
        )


class Beale:
    def __call__(self, x):
        x = mk_2d(x)
        x = 4.5 * x
        part1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        part2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        part3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return part1 + part2 + part3


class Branin:
    def __init__(self):
        self.a = 1
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

    def __call__(self, x):
        x = mk_2d(x)
        x1 = 7.5 * x[0] + 2.5
        x2 = 7.5 * x[1] + 7.5
        return (
            self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
            + self.s * (1 - self.t) * np.cos(x1)
            + self.s
        )


class Bukin:
    def __call__(self, x):
        x = mk_2d(x)
        x0 = x[0] * 5 - 10
        x1 = x[1] * 3
        return 100.0 * np.sqrt(np.abs(x1 - 0.01 * x0**2)) + 0.01 * np.abs(x0 + 10.0)


class CrossInTray:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 9 + 1
        x0 = x[0]
        x1 = x[1]
        part1 = (
            np.abs(
                np.sin(x0)
                * np.sin(x1)
                * np.exp(np.abs(100.0 - np.sqrt(x0**2 + x1**2) / np.pi))
            )
            + 1.0
        )
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


class DixonPrice:
    def __call__(self, x):
        x = x * 10
        part1 = (x[0] - 1) ** 2
        sum_terms = 0
        for i in range(2, len(x) + 1):
            xnew = x[i - 1]
            xold = x[i - 2]
            new = i * (2 * xnew**2 - xold) ** 2
            sum_terms += new
        return part1 + sum_terms


class EggHolder:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 511
        x1 = x[0]
        x2 = x[1]
        part1 = -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47.0))))
        return part1 + part2
