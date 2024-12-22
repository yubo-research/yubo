import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.models.exact_gp import ExactGP


class DUMBOGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    """
    To make an empty model, send torch.empty() tensors for train_x and train_y
     with the right dimensions, dtype, and device.
    """

    def __init__(self, train_X, train_Y):
        assert len(train_X) == len(train_Y), (len(train_X), len(train_Y))
        assert train_X.ndim == train_Y.ndim == 2, (train_X.ndim, train_Y.ndim)
        assert train_Y.shape[-1] == 1

        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)

        self._train_x = train_X
        self._train_Y = train_Y
        self.train_inputs = (train_X,)
        self.train_targets = train_Y.squeeze(-1)
        self._num_outputs = 1
        self._input_batch_shape = train_X.shape[:-2]

        self._eps_covar_diag = 1e-9

    def forward(self, X):
        # There might be two batch dimensions.
        # X ~ num_batch X num_joint X num_dim
        # X ~ num_batch_0 X num_batch_1 X num_joint X num_dim

        # Y ~ num_batch X num_joint X num_metrics

        if len(X.shape) == 2:
            b_nobatch = True
            X = X.unsqueeze(0)
        else:
            b_nobatch = False

        b, q, d = X.shape

        X_m = self._train_x
        Y_m = self._train_Y
        X = X.to(X_m)

        if self._train_x.numel() == 0:
            # to include X in the autograd graph
            mu = 0 * (X.sum(-1))
            vvar = 1 + 0 * (X.sum(-1))
            if b_nobatch:
                return MultivariateNormal(
                    mu.squeeze(0),
                    torch.diag_embed(vvar.squeeze(0)).to(X_m),
                )
            else:
                return MultivariateNormal(
                    mu,
                    torch.diag_embed(vvar).to(X_m),
                )

        distance = torch.cdist(X, X_m)
        # distance ~ num_batch x num_joint x num_train_x

        w = torch.exp(1.0 / (1e-1 + distance))
        assert torch.all(torch.isfinite(w)), (w, X)
        w = w / w.sum(dim=-1, keepdims=True)

        mu = (w * Y_m.squeeze(-1).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        vvar = (w * distance).sum(dim=-1)

        assert mu.shape == (b, q), (mu.shape, b, q)
        assert vvar.shape == (b, q), (vvar.shape, b, q)

        covar = torch.diag_embed(vvar + self._eps_covar_diag)
        assert covar.shape == (b, q, q), (covar.shape, b, q)

        if b_nobatch:
            mu = mu.squeeze(0)
            covar = covar.squeeze(0)

        return MultivariateNormal(mu, covar)
