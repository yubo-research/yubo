from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray

    def sample(self, num_samples):
        size = np.tile(self.se.expand_dims(-1), reps=num_samples)
        return self.mu + self.se * np.random.normal(size=size)


class EpsitemicNearestNeighbors:
    # TODO: train_YVar
    def __init__(self, train_x, train_y, k):
        assert len(train_x) == len(train_y), (len(train_x), len(train_y))
        assert train_x.ndim == train_y.ndim == 2, (train_x.ndim, train_y.ndim)

        self._train_x = train_x
        self._train_y = train_y
        self._num_obs, self._num_dim = self._train_x.shape
        self._num_metrics = self._train_y.shape[-1]
        self.k = k
        self._index = faiss.IndexFlatL2(train_x.shape[-1])
        self._index.add(train_x)
        self._eps_var = 1e-9

        # Maybe tune this on a sample of data
        #  if you want (somewhat) calibrated uncertainty estimates.
        self._var_scale = 1.0

    def add(self, x, y):
        self._index.add(x)
        self._train_x = np.append(self._train_x, x)
        self._train_y = np.append(self._train_y, y)

    def __len__(self):
        return self._index.ntotal

    def _idx_x_1(self, x):
        idx = np.where(np.all(self._train_x == x, axis=1))[0]
        if len(idx) > 1:
            # TODO: handle duplicates if needed
            idx = idx[[0]]
        elif len(idx) == 0:
            return None
        return idx

    def idx_x(self, x):
        idxs = [self._idx_x_1(x[i]) for i in range(x.shape[0])]
        idxs = np.array(idxs)
        assert len(idxs.flatten()) == x.shape[0]
        return idxs

    def about_neighbors(self, x, k=None):
        if k is None:
            k = self.k

        if len(self._train_x) == 0:
            return np.empty(shape=(0,), dtype=np.int64), np.empty(shape=(0,), dtype=np.float64)

        dists, idx = self._index.search(x, k=k)
        return idx.flatten(), dists.flatten()

    def neighbors(self, x, k=None):
        idx, _ = self.about_neighbors(x, k)
        return self._train_x[idx]

    def __call__(self, X, k=None):
        return self.posterior(X)

    def posterior(self, x, k=None):
        # X ~ num_batch X num_dim

        assert len(x.shape) == 2, ("NYI: Joint sampling", x.shape)
        b, d = x.shape
        assert d == self._num_dim, (d, self._num_dim)
        q = 1

        if self._train_x.shape[0] == 0:
            mu = 0 * (x.sum(-1))
            vvar = 1 + 0 * (x.sum(-1))
            return ENNNormal(
                mu.squeeze(0),
                np.sqrt(vvar.squeeze(0)),
            )

        dists, idx = self._index.search(x, k=self.k)
        mu = self._train_y[idx]
        assert mu.shape == (b, self.k, self._num_metrics), (mu.shape, b, self.k, self._num_metrics)
        vvar = np.expand_dims(dists, axis=-1)
        assert vvar.shape == (b, self.k, self._num_metrics), (vvar.shape, b, self.k, self._num_metrics)

        w = 1.0 / (self._eps_var + vvar)
        assert np.all(np.isfinite(w)), (w, x)
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
