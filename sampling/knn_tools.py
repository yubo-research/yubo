import numpy as np
from botorch.sampling.qmc import MultivariateNormalQMCEngine


def random_directions(num_samples, num_dim):
    import torch

    if False:
        u = np.random.normal(size=(num_samples, num_dim))
    else:
        u = (
            MultivariateNormalQMCEngine(
                mean=torch.zeros(size=(num_dim,)),
                cov=torch.diag(torch.ones(size=(num_dim,))),
            )
            .draw(n=num_samples)
            .detach()
            .numpy()
        )
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def nearest_neighbor(enn, x: np.array):
    num_samples, num_dim = x.shape

    dist_bdy = np.minimum(np.abs(1 - x).min(axis=1), np.abs(0 - x).min(axis=1))

    idx, dist = enn.about_neighbors(x, k=1)
    assert len(idx) == num_samples, (idx, dist)

    i = dist > dist_bdy
    idx[i] = -1
    dist[i] = dist_bdy[i]
    return idx, dist


def most_isolated(enn, x: np.ndarray):
    _, dists = nearest_neighbor(enn, x)
    i = np.where(dists == dists.max())[0]
    return x[i]


def farthest_neighbor(enn, x_0: np.ndarray, u: np.ndarray, eps_bound: float = 1e-6):
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

    idx_0 = enn.idx_x(x_0).flatten()
    if len(idx_0) != num_samples:
        assert False, "Can't find x_0 in training data for ENN"

    def _is_neighbor(x):
        return nearest_neighbor(enn, x)[0] == idx_0

    while (l_high - l_low).max() > eps_bound:
        l_mid = (l_low + l_high) / 2
        x_mid = x_0 + l_mid * u
        x_mid = np.maximum(0, np.minimum(1, x_mid))
        a = _is_neighbor(x_mid)

        l_low[a] = l_mid[a]
        l_high[~a] = l_mid[~a]

    x = x_0 + l_low * u
    x = np.maximum(0, np.minimum(1, x))
    # assert np.all(_is_neighbor(x)), (idx_0, enn.about_neighbors(x, k=1))
    return x
