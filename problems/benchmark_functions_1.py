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
from .benchmark_functions_1d import (  # noqa: F401
    Beale,
    Branin,
    Bukin,
    CrossInTray,
    DixonPrice,
    DropWave,
    EggHolder,
)

# Requirements:
# - x in [-1,1]**num_dim
# - have *minima* [not maxima]
# - support any number of dimensions as input


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
        return -self.a * np.exp(-self.b * np.sqrt((x**2).mean())) - np.exp(np.cos(self.c * x).mean()) + self.a + np.e
