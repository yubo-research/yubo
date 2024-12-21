import torch
from gpytorch.distributions import MultivariateNormal


class DUMBOGP:
    """
    To make an empty model, send torch.empty() tensors for train_x and train_y
     with the right dimensions, dtype, and device.
    """

    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y), (len(train_x), len(train_y))

        self._train_x = train_x.unsqueeze(0)
        self._train_y = train_y.unsqueeze(0)
        self.train_inputs = (train_x,)
        self.train_targets = train_y.unsqueeze(0)
        self.num_outputs = self._train_y.shape[-1]

    def __call__(self, X):
        return self.posterior(X)

    def posterior(self, X, posterior_transform=None):
        # X ~ num_batch X num_joint X num_dim
        # Y ~ num_batch X num_joint X num_metrics

        if len(X.shape) == 2:
            X = X.unsqueeze(0)

        b, q, d = X.shape
        m = self._train_y.shape[-1]

        X_m = self._train_x
        Y_m = self._train_y

        if self._train_x.numel() == 0:
            return MultivariateNormal(
                torch.zeros(size=(m, b, q)).to(X_m),
                torch.diag_embed(torch.ones(size=(m, b, q))).to(X_m),
            )

        X = X.to(X_m)

        distance = torch.cdist(X, X_m)
        # distance ~ num_batch x num_joint x num_train_x

        if False:
            w = torch.exp(1.0 / (1e-2 + distance))
            assert torch.all(torch.isfinite(w))

            w = w / w.sum(dim=-1, keepdims=True)

            # w ~ num_batch x num_joint x num_train_x
            # Y_m ~ 1 x num_train_x x num_metrics

            mu = (w.unsqueeze(-1) * Y_m.unsqueeze(1)).sum(dim=-2)

        else:
            tm = torch.min(distance, dim=-1)
            # i ~ num_batch x num_joint;  indexes num_train_x
            i = tm.indices
            assert i.shape == (b, q), (i.shape, b, q)
            Y_m = Y_m.squeeze(0).swapdims(0, 1).unsqueeze(-1)
            mu = torch.cat([Y_m[:, i[:, ii]] for ii in range(i.shape[-1])], dim=-1)

        # mu ~ num_metrics X num_batch X num_joint
        assert mu.shape == (m, b, q), (mu.shape, m, b, q)

        var = torch.tile(torch.amin(distance, dim=-1), (m, 1, 1))
        assert var.shape == (m, b, q), (var.shape, m, b, q)

        covar = torch.diag_embed(var + 1e-9)
        assert covar.shape == (m, b, q, q), (covar.shape, m, b, q)

        return MultivariateNormal(mu, covar)
