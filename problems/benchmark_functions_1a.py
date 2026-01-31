import numpy as np

from .benchmark_util import mk_2d, mk_4d


class Griewank:
    def __call__(self, x):
        x = x * 600
        part1 = np.sum(x**2 / 4000.0)
        num_dim = len(x)
        part2 = np.prod(np.cos(x / np.sqrt((1 + np.arange(num_dim)))))
        return part1 - part2 + 1


class GrLee12:
    def __call__(self, x):
        x = mk_2d(x)
        x = x[0] + 1.5
        return np.sin(10.0 * np.pi * x) / (2.0 * x) + (x - 1.0) ** 4


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


class HolderTable:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 10
        x0 = x[0]
        x1 = x[1]
        part1 = np.sin(x0) * np.cos(x1)
        part2 = np.exp(np.abs(1 - np.sqrt(x0**2 + x1**2) / np.pi))
        return -np.abs(part1 * part2)


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


class Michalewicz:
    m = 10

    def __call__(self, x):
        x = np.pi * (1 + np.asarray(x)) / 2
        d = len(x)
        indices = np.arange(1, d + 1)
        sine_term1 = np.sin(x)
        sine_term2 = np.sin(indices * x**2 / np.pi) ** (2 * self.m)
        return -np.sum(sine_term1 * sine_term2)


class Powell:
    def __call__(self, x):
        x = 0.5 + 4.5 * np.asarray(x)
        d = len(x)
        if d % 4 != 0:
            x = mk_4d(x)
            d = len(x)
        x_grouped = x.reshape(-1, 4)
        term1 = (x_grouped[:, 0] + 10 * x_grouped[:, 1]) ** 2
        term2 = 5 * (x_grouped[:, 2] - x_grouped[:, 3]) ** 2
        term3 = (x_grouped[:, 1] - 2 * x_grouped[:, 2]) ** 4
        term4 = 10 * (x_grouped[:, 0] - x_grouped[:, 3]) ** 4
        return np.sum(term1 + term2 + term3 + term4)


class Rastrigin:
    def __call__(self, x):
        x = x * 5.12
        num_dim = len(x)
        return 10 + np.sum(x**2 - 10 * np.cos(np.pi * 2 * x)) / num_dim


class Rosenbrock:
    def __call__(self, x):
        x = x * 2.048
        part = 0
        num_dim = len(x)
        for i in range(num_dim - 1):
            part += (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return 100.0 * part / num_dim


class Shubert:
    def __call__(self, x):
        x = mk_2d(x)
        x = x * 5.12
        x0 = x[0]
        x1 = x[1]
        part1 = 0
        part2 = 0
        for i in range(1, 6):
            new1 = i * np.cos((i + 1) * x0 + i)
            new2 = i * np.cos((i + 1) * x1 + i)
            part1 += new1
            part2 += new2
        return part1 * part2
