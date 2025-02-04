import numpy as np

from .exceptions import WrongDimensions


class Alpine:
    def __call__(self, x):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


class Easom:
    def __call__(self, x):
        if len(x) != 2:
            raise WrongDimensions()
        y = x[1]
        x = x[0]
        return np.sum(-np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2) - (y - np.pi) ** 2))


class Booth:
    def __call__(self, x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


class Himmelblau:
    def __call__(self, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class Matyas:
    def __call__(self, x):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


class Zettl:
    def __call__(self, x):
        return (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]


class Sum_Squares:
    def __call__(self, x):
        x = np.array(x)
        return np.sum((x**2) * np.arange(1, 1 + 1))


class Perm:
    beta = 10

    def __call__(self, x):
        n = 2
        return sum(sum((j + self.beta) * (x[j] ** i - 1) for j in range(n)) ** 2 for i in range(1, n + 1))


class Salomon:
    def __call__(self, x):
        r = np.sqrt(sum(xi**2 for xi in x))
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r


class Whitley:
    def __call__(self, x):
        x = np.array(x)
        n = len(x)
        sum1 = np.sum((x**2 - 10) ** 2)
        sum2 = np.sum(x**2)
        return (10 * n + sum1 + 1) / (30 * n + sum2)


class Brown:
    def __call__(self, x):
        x = np.array(x)
        sum1 = np.sum(x**2 - 10) ** 2
        sum2 = np.prod(x**2)
        return sum1 + sum2


class Zakharov:
    def __call__(self, x):
        x = np.array(x)
        term1 = np.sum(x**2)
        term2 = np.sum((0.5 * np.arange(1, 2 + 1) * x) ** 2)
        return term1 + term2
