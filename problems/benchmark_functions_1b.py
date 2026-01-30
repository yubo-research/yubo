import numpy as np

from .benchmark_util import mk_2d, mk_4d


class Shekel:
    def __init__(self):
        self.beta = 0.1 * np.array((1, 2, 2, 4, 4, 6, 3, 7, 5, 5)).T
        self.C = np.array(
            (
                4,
                1,
                8,
                6,
                3,
                2,
                5,
                8,
                6,
                7,
                4,
                1,
                8,
                6,
                7,
                9,
                3,
                1,
                2,
                3.6,
                4,
                1,
                8,
                6,
                3,
                2,
                5,
                8,
                6,
                7,
                4,
                1,
                8,
                6,
                7,
                9,
                3,
                1,
                2,
                3.6,
            )
        ).reshape(4, 10)

    def __call__(self, x):
        if len(x) != 4:
            x = mk_4d(x)
        x = x * 5 + 5
        m = self.C.shape[1]
        outer = 0
        for i in range(m):
            bi = self.beta[i]
            inner = 0
            for j in range(4):
                inner += (x[j] - self.C[j][i]) ** 2
            outer += 1 / (inner + bi)
        return outer


class SixHumpCamel:
    def __call__(self, x):
        x = mk_2d(x)
        x0 = x[0] * 3
        x1 = x[1] * 2
        return (4 - 2.1 * x0**2 + x0**4 / 3) * x0**2 + x0 * x1 + (4 * x1**2 - 4) * x1**2


class StybTang:
    def __call__(self, x):
        x = x * 5
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


class ThreeHumpCamel:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 5
        x0 = x[0]
        x1 = x[1]
        return 2.0 * x0**2 - 1.05 * x0**4 + x0**6 / 6.0 + x0 * x1 + x1**2


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


class Bohachevsky1:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 100 * x
        x1, x2 = x[0], x[1]
        term1 = x1**2
        term2 = 2 * x2**2
        term3 = -0.3 * np.cos(3 * np.pi * x1)
        term4 = -0.4 * np.cos(4 * np.pi * x2)
        return term1 + term2 + term3 + term4 + 0.7


class RotatedHyperEllipsoid:
    def __call__(self, x):
        x = np.asarray(x)
        x = 65.536 * x
        return np.sum(np.cumsum(x**2))


class SumOfDifferentPowers:
    def __call__(self, x):
        x = np.asarray(x)
        return np.sum(np.abs(x) ** (np.arange(1, len(x) + 1) + 1))


class Trid:
    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = (d**2) * x
        term1 = np.sum((x - 1) ** 2)
        term2 = np.sum(x[1:] * x[:-1])
        return term1 - term2
