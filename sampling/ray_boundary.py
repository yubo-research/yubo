import torch


def ray_boundary(x, u, eps=1e-6):
    assert x.dtype == torch.double
    # x ~ num_batch X num_dim
    # u ~ num_batch X num_dim

    # x_ray = x + t*u
    t_min = (0 - x) / u
    t_max = (1 - x) / u
    # one t_min (t_max) for each coordinate, each x in batch

    t_candidates = torch.where(u > 0, t_max, torch.where(u < 0, t_min, torch.inf))
    t_min_positive = torch.min(t_candidates[t_candidates > 0])

    x = x + t_min_positive * u
    assert x.min() > -eps, (x.min(), x)
    assert x.max() < 1 + eps, (x.max(), x)

    return torch.maximum(torch.tensor(0.0), torch.minimum(torch.tensor(1.0), x))
