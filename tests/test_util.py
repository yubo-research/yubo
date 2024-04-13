def gp_parabola(*, num_samples=3, num_dim=2):
    import torch

    from acq.fit_gp import fit_gp_XY

    X_0 = 0.3 * torch.ones(size=(num_dim,))
    X = torch.rand(size=torch.Size([num_samples, num_dim]), dtype=torch.double)

    Y = -((X - X_0) ** 2).sum(dim=1)
    Y = Y + 0.25 * torch.randn(size=Y.shape)
    Y = Y[:, None]

    return fit_gp_XY(X, Y), X_0
