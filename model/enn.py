from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray

    def sample(self):
        return self.mu + self.se * np.random.normal(size=self.se.shape)


class EpsitemicNearestNeighbors:
    # TODO: train_YVar
    def __init__(self, train_X, train_Y, k):
        assert len(train_X) == len(train_Y), (len(train_X), len(train_Y))
        assert train_X.ndim == train_Y.ndim == 2, (train_X.ndim, train_Y.ndim)

        self._train_X = train_X
        self._train_Y = train_Y
        self._num_obs, self._num_dim = self._train_X.shape
        self._num_metrics = self._train_Y.shape[-1]
        self.k = k
        self._index = faiss.IndexFlatL2(train_X.shape[-1])
        self._index.add(train_X)
        self._eps_var = 1e-9

        # Maybe tune this on a sample of data
        #  if you want calibrated uncertainty estimates.
        self._var_scale = 1.0

    def __call__(self, X):
        # X ~ num_batch X num_dim

        assert len(X.shape) == 2, ("NYI: Joint sampling", X.shape)
        b, d = X.shape
        assert d == self._num_dim, (d, self._num_dim)
        q = 1

        if self._train_X.shape[0] == 0:
            mu = 0 * (X.sum(-1))
            vvar = 1 + 0 * (X.sum(-1))
            return ENNNormal(
                mu.squeeze(0),
                np.sqrt(vvar.squeeze(0)),
            )

        dists, idx = self._index.search(X, k=self.k)
        mu = self._train_Y[idx]
        assert mu.shape == (b, self.k, self._num_metrics), (mu.shape, b, self.k, self._num_metrics)
        vvar = np.expand_dims(dists, axis=-1)
        assert vvar.shape == (b, self.k, self._num_metrics), (vvar.shape, b, self.k, self._num_metrics)

        w = 1.0 / (self._eps_var + vvar)
        assert np.all(np.isfinite(w)), (w, X)
        norm = w.sum(axis=1)
        # sum over k neighbors
        mu = (w * mu).sum(axis=1) / norm
        vvar = 1.0 / norm

        assert mu.shape == (b, q), (mu.shape, b, q)
        # TODO: include self variance (Yvar) in 1 / sum(1/var)
        vvar = self._var_scale * vvar
        assert vvar.shape == (b, q), (vvar.shape, b, q)
        vvar = np.maximum(self._eps_var, vvar)

        return ENNNormal(mu, np.sqrt(vvar))
