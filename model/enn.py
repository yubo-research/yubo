import faiss
import torch
from gpytorch.distributions import MultivariateNormal


class EpsitemicNearestNeighbors:
    # TODO: train_YVar
    def __init__(self, train_X, train_Y, k, fancy=True):
        assert len(train_X) == len(train_Y), (len(train_X), len(train_Y))
        assert train_X.ndim == train_Y.ndim == 2, (train_X.ndim, train_Y.ndim)

        self._train_X = train_X
        self._train_Y = train_Y
        self._num_obs, self._num_dim = self._train_X.shape
        self._num_metrics = self._train_Y.shape[-1]
        self.k = k
        self._index = faiss.IndexFlatL2(train_X.shape[-1])
        self._index.add(train_X)
        self._eps_covar_diag = torch.tensor(1e-9)

        self.fancy = fancy
        # Maybe tune this on a sample of data
        #  if you want calibrated uncertainty estimates.
        self._var_scale = 1.0

    def __call__(self, X):
        # X ~ num_batch X num_dim

        assert len(X.shape) == 2, "NYI: Joint sampling"
        b, d = X.shape
        assert d == self._num_dim, (d, self._num_dim)
        q = 1

        X = X.to(self._train_X)

        if self._train_X.numel() == 0:
            mu = 0 * (X.sum(-1))
            vvar = 1 + 0 * (X.sum(-1))
            return MultivariateNormal(
                mu.squeeze(0),
                torch.diag_embed(vvar.squeeze(0)).to(self._train_X),
            )

        dists, idx = self._index.search(X, k=self.k)
        dists = torch.tensor(dists)
        mu = self._train_Y[idx]
        assert mu.shape == (b, self.k, self._num_metrics), (mu.shape, b, self.k, self._num_metrics)
        vvar = dists.unsqueeze(-1)
        assert vvar.shape == (b, self.k, self._num_metrics), (vvar.shape, b, self.k, self._num_metrics)

        if not self.fancy:
            mu = mu.mean(dim=1)
            vvar = vvar.mean(dim=1)
        else:
            w = 1.0 / vvar
            assert torch.all(torch.isfinite(w)), (w, X)
            norm = w.sum(dim=1)
            # sum over k neighbors
            print(w.shape, norm.shape, mu.shape)
            mu = (w * mu).sum(dim=1) / norm
            vvar = 1.0 / norm

        assert mu.shape == (b, q), (mu.shape, b, q)
        vvar = self._var_scale * vvar
        assert vvar.shape == (b, q), (vvar.shape, b, q)
        vvar = torch.maximum(self._eps_covar_diag, vvar)
        covar = torch.diag_embed(vvar)
        assert covar.shape == (b, q, q), (covar.shape, b, q)

        return MultivariateNormal(mu, covar)
