import numpy as np
from botorch.sampling.qmc import MultivariateNormalQMCEngine


def target_directions(x_0: np.ndarray):
    x_t = np.random.uniform(size=x_0.shape)
    u = x_t - x_0
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def approx_ard(x_max, y_max, x_neighbors, y_neighbors, eps=0.01):
    dx = x_max - x_neighbors
    slope = np.abs(np.maximum(0.0, y_max - y_neighbors)[:, None] / np.maximum(eps, dx))
    weight = np.linalg.norm(x_max - x_neighbors, axis=1)
    weight = weight / weight.sum()
    u = (weight[:, None] * slope).sum(axis=0)
    u_norm = np.linalg.norm(u)

    if u_norm > 0:
        return u / u_norm
    return np.zeros_like(x_max)


def random_directions(num_samples, num_dim):
    import torch

    if True:
        u = np.random.normal(size=(num_samples, num_dim))
    else:
        for _ in range(5):
            u = (
                MultivariateNormalQMCEngine(
                    mean=torch.zeros(size=(num_dim,)),
                    cov=torch.diag(torch.ones(size=(num_dim,))),
                )
                .draw(n=num_samples)
                .detach()
                .numpy()
            )
            if not np.any(np.isnan(u)):
                break

    assert not np.any(np.isnan(u))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def nearest_neighbor(enn, x: np.array, p_boundary_is_neighbor):
    num_samples, num_dim = x.shape

    idx, dist = enn.about_neighbors(x, k=1)
    idx = idx.flatten()
    dist = dist.flatten()
    assert len(idx) == num_samples, (idx, dist)

    i_bin = np.where(np.random.binomial(n=1, p=p_boundary_is_neighbor, size=num_samples))[0]
    if len(i_bin) > 0:
        dist_bdy = np.minimum(np.abs(1 - x[i_bin]).min(axis=1), np.abs(0 - x[i_bin]).min(axis=1))
        i = np.where(dist[i_bin] > dist_bdy)[0]
        i_bin = i_bin[i]
        dist_bdy = dist_bdy[i]
        idx[i_bin] = -1
        dist[i_bin] = dist_bdy

    return idx, dist


def most_isolated(enn, x: np.ndarray, p_boundary_is_neighbor):
    _, dists = nearest_neighbor(enn, x, p_boundary_is_neighbor=p_boundary_is_neighbor)
    i = np.where(dists == dists.max())[0]
    return x[i]


def farthest_neighbor(enn, x_0: np.ndarray, u: np.ndarray, eps_bound: float = 1e-6, p_boundary_is_neighbor=0.0):
    """
    Find the farthest point from x_0 along direction u that still has x_0 as its nearest neighbor.
    Alternatively, find the boundary of the Vornoi cell that has x_0 as its center.

    Assumes a bounding box at [0,1]^num_dim, which counts as a "neighbor".

    x_0: num_samples x num_dim, center of Voronoi cell
    u: num_samples x num_dim, unit-length direction vectors
    """

    assert not np.any(np.isnan(x_0)), x_0
    assert len(x_0.shape) == 2, x_0.shape
    num_samples, num_dim = x_0.shape

    l_low = np.zeros(shape=(num_samples, 1))
    l_high = 2 * np.sqrt(num_dim) * np.ones(shape=(num_samples, 1))

    
    idx_0 = enn.idx_x(x_0).flatten()
    if len(idx_0) != num_samples:
        assert False, "Can't find x_0 in training data for ENN"

    def _is_neighbor(x):
        return nearest_neighbor(enn, x, p_boundary_is_neighbor=p_boundary_is_neighbor)[0] == idx_0

    while (l_high - l_low).max() > eps_bound:
        assert not np.any(np.isnan(l_low)), l_low
        assert not np.any(np.isnan(l_high)), l_high

        l_mid = (l_low + l_high) / 2
        x_mid = x_0 + l_mid * u
        x_mid = np.maximum(0, np.minimum(1, x_mid))
        a = _is_neighbor(x_mid)

        l_low[a] = l_mid[a]
        l_high[~a] = l_mid[~a]

    x = x_0 + l_low * u
    x = np.maximum(0, np.minimum(1, x))
    assert not np.any(np.isnan(x)), x
    # assert np.all(_is_neighbor(x)), (idx_0, enn.about_neighbors(x, k=1))
    return x
