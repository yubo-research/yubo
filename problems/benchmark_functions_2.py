import numpy as np

from .benchmark_util import mk_2d

# Requirements:
# - x in [-1,1]**num_dim
# - have *minima* [not maxima]
# - support any number of dimensions as input


class Alpine:
    """
    See: https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/162-alpine-function-no-1
    """

    def __call__(self, x):
        x = 5 * (1 + np.asarray(x))
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


class Easom:
    """
    See: https://www.sfu.ca/~ssurjano/easom.html
    """

    def __call__(self, x):
        x = 100 * mk_2d(x)
        y = x[1]
        x = x[0]
        return np.sum(-np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2) - (y - np.pi) ** 2))


class Booth:
    def __call__(self, x):
        x = 10 * mk_2d(x)
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


class Himmelblau:
    def __call__(self, x):
        x = 6 * mk_2d(x)
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class Matyas:
    def __call__(self, x):
        x = 10 * mk_2d(x)
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


class Zettl:
    def __call__(self, x):
        x = 2 + 3 * mk_2d(x)
        return (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]


class Sum_Squares:
    def __call__(self, x):
        x = 10 * np.asarray(x)
        return np.sum((x**2) * np.arange(1, len(x) + 1))


class Perm:
    beta = 10

    def __call__(self, x):
        d = len(x)
        x = d * x
        return np.sum([np.sum([((1 + j) ** (1 + i) + self.beta) * ((x[j] / (1 + j)) ** (1 + i) - 1) for j in range(d)]) ** 2 for i in range(d)])


class Salomon:
    def __call__(self, x):
        x = 100 * np.asarray(x)
        r = np.linalg.norm(x)
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r


class Whitley:
    """
    See: http://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Whitley
    """

    def __call__(self, x):
        x = 10.24 * np.asarray(x)
        d = len(x)

        i_indices, j_indices = np.meshgrid(range(d), range(d), indexing="ij")
        xi_squared = x[i_indices] ** 2
        xj = x[j_indices]

        term1 = (100 * (xi_squared - xj) ** 2 + (1 - xj) ** 2) ** 2 / 4000
        term2 = -np.cos(100 * (xi_squared - xj) ** 2 + (1 - xj) ** 2)

        return np.sum(term1 + term2 + 1)


class Brown:
    """
    See: https://www.indusmic.com/post/brown-function#:~:text=Brown%20Function%20is%20usually%20evaluated,This%20function%20is%20smooth.&text=Brown%20Function%20is%20a%20unimodal,defined%20on%20n%2D%20dimensional%20space.
    """

    def __call__(self, x):
        x = 1.5 + 2.5 * np.asarray(x)
        d = len(x)
        if d < 2:
            x = mk_2d(x)
            d = 2

        xi_squared = x[:-1] ** 2  # x_i^2 for i = 1 to n-1
        xi_plus_1_squared = x[1:] ** 2  # x_i+1^2 for i = 1 to n-1

        term1 = xi_squared * (xi_plus_1_squared + 1)
        term2 = xi_plus_1_squared * (xi_squared + 1)

        return np.sum(term1 + term2)


class Zakharov:
    """
    See: https://www.sfu.ca/~ssurjano/zakharov.html
    """

    def __call__(self, x):
        x = 2.5 + 7.5 * np.array(x)
        s = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return np.sum(x**2) + s**2 + s**4
