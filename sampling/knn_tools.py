import numpy as np


def random_direction(num_dim):
    u = np.random.normal(size=(num_dim,))
    return u / np.linalg.norm(u)


def _idx_nearest_neighbor(enn, x: np.array):
    num_dim = x.shape[-1]
    dists_bdy = []
    for i in range(num_dim):
        dists_bdy.append(np.abs(1 - x))
        dists_bdy.append(np.abs(-1 - x))
    dist_bdy = np.min(dists_bdy)

    idx, dist = enn.about_neighbors(x, k=1)
    if len(idx) != 1:
        breakpoint()
    assert len(idx) == 1, (idx, dist)

    if dist[0] < dist_bdy:
        return idx[0]
    return -1


def farthest_neighbor(enn, x_0: np.array, u: np.array, eps_bound: float = 1e-6):
    """
    Find the farthest point from x_0 along direction u that still has x_0 as its nearest neighbor.
    Alternatively, find the boundary of the Vornoi cell that has x_0 as its center.

    Assumes a bounding box at [0,1]^num_dim, which counts as a "neighbor".

    x_0: num_samples x num_dim, center of Voronoi cell
    u: num_samples x num_dim, unit-length direction vectors
    """
    assert len(x_0.shape) == 2, x_0.shape
    num_samples = x_0.shape[0]
    l_low = np.zeros(shape=(num_samples, 1))
    l_high = np.ones(shape=(num_samples, 1))

    idx_0 = enn.idx_x(x_0)
    assert len(idx_0) == 1

    def _is_neighbor(x):
        return _idx_nearest_neighbor(enn, x) == idx_0

    while (l_high - l_low).max() > eps_bound:
        l_mid = (l_low + l_high) / 2
        x_mid = x_0 + l_mid * u
        x_mid = np.maximum(-1, np.minimum(1, x_mid))
        a = _is_neighbor(x_mid)
        l_low[a] = l_mid[a]
        l_high[~a] = l_mid[~a]

    return (l_low + l_high) / 2
