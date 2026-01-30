import numpy as np

from .benchmark_util import mk_2d, mk_4d


class Perm:
    def __init__(self, beta=0.5):
        self.beta = beta

    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = d * x
        total = 0.0
        for i in range(1, d + 1):
            inner = 0.0
            for j in range(1, d + 1):
                term = (j + self.beta) * (x[j - 1] ** i - 1 / (j**i))
                inner += term
            total += inner**2
        return total


class Schaffer2:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 100 * x
        x1, x2 = x[0], x[1]
        numerator = np.sin(x1**2 - x2**2) ** 2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        return 0.5 + numerator / denominator


class Schaffer4:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 100 * x
        x1, x2 = x[0], x[1]
        diff = np.abs(x1**2 - x2**2)
        numerator = np.cos(np.sin(diff)) ** 2 - 0.5
        denominator = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        return 0.5 + numerator / denominator


class Schwefel:
    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = 500 * x
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class McCormick:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x1 = (4 + 1.5) / 2 * x[0] + (4 - 1.5) / 2
        x2 = (4 + 3) / 2 * x[1] + (4 - 3) / 2
        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        return term1 + term2 + term3 + term4 + 1


class PowerSum:
    def __init__(self, b=None):
        self.b = np.array(b) if b is not None else np.array([8, 18, 44, 114])

    def __call__(self, x):
        x = np.asarray(x)
        x = mk_4d(x)
        d = len(x)
        x = (x + 1) * d / 2
        total = 0.0
        for i in range(1, d + 1):
            inner_sum = np.sum(x**i)
            total += (inner_sum - self.b[i - 1]) ** 2
        return total


class PermDBeta:
    def __init__(self, beta=0.5):
        self.beta = beta

    def __call__(self, x):
        x = np.asarray(x)
        d = len(x)
        x = d * x
        total = 0.0
        for i in range(1, d + 1):
            inner = 0.0
            for j in range(1, d + 1):
                term = (j**i + self.beta) * (((x[j - 1] / j) ** i) - 1)
                inner += term
            total += inner**2
        return total


class GoldsteinPrice:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 2 * x
        x1, x2 = x[0], x[1]
        part1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        part2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        return part1 * part2


class Colville:
    def __call__(self, x):
        x = np.asarray(x)
        x = mk_4d(x)
        x = 10 * x
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        term1 = 100 * (x1**2 - x2) ** 2
        term2 = (x1 - 1) ** 2
        term3 = 90 * (x3**2 - x4) ** 2
        term4 = (x3 - 1) ** 2
        term5 = 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        term6 = 19.8 * (x2 - 1) * (x4 - 1)
        return term1 + term2 + term3 + term4 + term5 + term6


class DeJong5:
    def __init__(self):
        a_vals = np.array([-32, -16, 0, 16, 32])
        self.a = np.array([[x, y] for x in a_vals for y in a_vals]).T

    def __call__(self, x):
        x = np.asarray(x)
        x = mk_2d(x)
        x = 65.536 * x
        x1, x2 = x[0], x[1]
        sum_terms = 0.0
        for i in range(25):
            ai1 = self.a[0, i]
            ai2 = self.a[1, i]
            denom = i + 1 + (x1 - ai1) ** 6 + (x2 - ai2) ** 6
            sum_terms += 1 / denom
        return 1.0 / (0.002 + sum_terms)
