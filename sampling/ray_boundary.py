import numpy as np
import torch


def ray_boundary(x_0, u, eps=1e-6):
    assert x_0.dtype == torch.double, x_0.dtype
    # x ~ num_batch X num_dim
    # u ~ num_batch X num_dim

    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)

    def _bound(xx):
        return torch.maximum(zero, torch.minimum(one, xx))

    x_0 = _bound(x_0)
    # x_ray = x + t*u
    t_min = (zero - x_0) / u
    t_max = (one - x_0) / u
    # one t_min (t_max) for each coordinate, each x in batch

    t_candidates = torch.where(u > 0, t_max, torch.where(u < 0, t_min, torch.inf))
    t_min_positive = torch.min(t_candidates[t_candidates >= 0])

    x = x_0 + t_min_positive * u
    assert x.min() > -eps, (x.min(), x, x_0)
    assert x.max() < 1 + eps, (x.max(), x, x_0)

    return _bound(x)


def ray_boundary_np(x_0, u, eps=1e-6):
    assert x_0.dtype == np.float64, x_0.dtype
    # x ~ num_batch X num_dim
    # u ~ num_batch X num_dim

    zero = 0.0
    one = 1.0

    def _bound(xx):
        return np.maximum(zero, np.minimum(one, xx))

    x_0 = _bound(x_0)
    # x_ray = x + t*u
    t_min = (zero - x_0) / u
    t_max = (one - x_0) / u
    # one t_min (t_max) for each coordinate, each x in batch

    t_candidates = np.where(u > 0, t_max, np.where(u < 0, t_min, np.inf))
    t_candidates = t_candidates[t_candidates >= 0]
    t_min_positive = np.min(t_candidates)

    x = x_0 + t_min_positive * u
    assert x.min() > -eps, (x.min(), x, x_0)
    assert x.max() < 1 + eps, (x.max(), x, x_0)

    return _bound(x)
