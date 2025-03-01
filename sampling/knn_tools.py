import numpy as np


def random_direction(num_dim):
    u = np.random.normal(size=(num_dim,))
    return u / np.linalgnorm(u)


# def _nearest_neighbor(enn, x: np.array):
#     num_dim = x.shape[-1]
#     dists_bdy = []
#     for i in range(num_dim):
#         dists_bdy.append(np.abs(1 - x))
#         dists_bdy.append(np.abs(-1 - x))
#     dist_bdy = np.min(dists_bdy)


def farthest_neighbor(enn, x: np.array, u: np.array, eps_bound: float):
    # TODO: Find farthest point on the ray (x,u) that
    #  still has x as its nearest neighbor.
    #
    # - bisection search
    # - boundary counts as a neighbor

    """
    Bisection search along u until Voronoi bound is found within eps_bound
    Assumes a bounding box [0,1]^num_dim, which counts as a "neighbor".

    X: num_samples x num_dim, starting point
    u: num_samples x num_dim, unit-length direction vectors
    """
    num_samples = x.shape[0]
    l_low = np.zeros(shape=(num_samples, 1))
    l_high = np.ones(shape=(num_samples, 1))

    # lb, ub = 0, 1

    def _accept(X):
        assert False, "NYI: X"
        pass

    while (l_high - l_low).max() > eps_bound:
        l_mid = (l_low + l_high) / 2
        x_mid = x + l_mid * u
        a = _accept(x_mid)
        l_low[a] = l_mid[a]
        l_high[~a] = l_mid[~a]

    return (l_low + l_high) / 2
