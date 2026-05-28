from __future__ import annotations

from types import SimpleNamespace

import torch


def fake_gp_multivariate_normal(x: torch.Tensor):
    import gpytorch.distributions as gd
    from gpytorch.lazy import lazify

    n = int(x.shape[0])
    m = torch.zeros(n, dtype=x.dtype, device=x.device)
    c = torch.eye(n, dtype=x.dtype, device=x.device) * 0.1 + torch.eye(n, dtype=x.dtype, device=x.device) * 1e-4
    return gd.MultivariateNormal(m, lazify(c))


def make_fake_gp():
    gp = SimpleNamespace(
        covar_module=SimpleNamespace(base_kernel=SimpleNamespace(lengthscale=torch.ones(1, 2, dtype=torch.float64))),
        likelihood=SimpleNamespace(__call__=lambda mv: mv),
    )
    gp.to = lambda dtype=None, device=None: gp
    gp.__call__ = fake_gp_multivariate_normal
    return gp
