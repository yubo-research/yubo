from dataclasses import dataclass

import faiss
import numpy as np
import torch


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray

    def sample(self, num_samples, clip=None):
        if isinstance(num_samples, torch.Size):
            num_samples = list(num_samples)
            assert len(num_samples) == 1, num_samples
            num_samples = num_samples[0]
        size = list(self.se.shape)
        size.append(num_samples)

        eps = np.random.normal(size=size)
        if clip is not None:
            eps = np.clip(eps, a_min=-clip, a_max=clip)

        return np.expand_dims(self.mu, -1) + np.expand_dims(self.se, -1) * eps


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
        self._lookup = None

    def add(self, x, y):
        self._index.add(x)
        self._train_x = np.append(self._train_x, x)
        self._train_y = np.append(self._train_y, y)
        if self._lookup is not None:
            assert False, "NYI: Add to lookup"

    def calibrate(self, var_scale):
        self._var_scale = var_scale

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

    def idx_x_slow(self, x):
        # Loop!
        idxs = [self._idx_x_1(x[i]) for i in range(x.shape[0])]
        idxs = np.array(idxs)
        assert len(idxs.flatten()) == x.shape[0]
        return idxs

    def idx_x(self, x):
        if self._lookup is None:
            train_x_view = self._train_x.view([("", self._train_x.dtype)] * self._train_x.shape[1])
            self._lookup = {tuple(row.tolist()): i for i, row in enumerate(train_x_view)}

        x_view = x.view([("", x.dtype)] * x.shape[1])
        idx = np.array([self._lookup[tuple(row.tolist())] for row in x_view], dtype=int)

        return idx

    def idx_fast(self, x):
        idx, dist = self.about_neighbors(x, k=1)
        i = np.where(dist > 0)[0]
        if len(i) > 1e-4:
            print(f"WRAN: {len(i)} points may not be in training data, max(dist) = {dist.max()}")
        return idx.flatten()

    def about_neighbors(self, x, k=None):
        if k is None:
            k = self.k

        if len(self._train_x) == 0:
            return np.empty(shape=(0,), dtype=np.int64), np.empty(shape=(0,), dtype=np.float64)

        dists, idx = self._index.search(x, k=k)
        return idx, dists

    def neighbors(self, x, k=None):
        idx, _ = self.about_neighbors(x, k)
        return self._train_x[idx]

    def __call__(self, X):
        return self.posterior(
            X,
        )

    def posterior(self, x, k=None, exclude_nearest=False):
        if k is None:
            k = self.k

        # X ~ num_batch X num_dim
        x = np.array(x)

        assert len(x.shape) == 2, ("NYI: Joint sampling", x.shape)
        b, d = x.shape
        assert d == self._num_dim, (d, self._num_dim)

        if self._train_x.shape[0] == 0:
            mu = 0 * (x.sum(-1))
            vvar = 1 + 0 * (x.sum(-1))
            return ENNNormal(
                mu.squeeze(0),
                np.sqrt(vvar.squeeze(0)),
            )

        if exclude_nearest:
            dists, idx = self._index.search(x, k=k + 1)
            dists = dists[:, 1:]
            idx = idx[:, 1:]
        else:
            dists, idx = self._index.search(x, k=k)

        return self._calc_enn_normal(b, dists, idx, k)

    def _calc_enn_normal(self, batch_size, dists, idx, k):
        q = 1

        mu = self._train_y[idx]
        assert mu.shape == (batch_size, k, self._num_metrics), (mu.shape, batch_size, k, self._num_metrics)
        vvar = np.expand_dims(dists, axis=-1)
        assert vvar.shape == (batch_size, k, self._num_metrics), (vvar.shape, batch_size, k, self._num_metrics)

        w = 1.0 / (self._eps_var + vvar)
        assert np.all(np.isfinite(w)), w
        norm = w.sum(axis=1)
        # sum over k neighbors
        mu = (w * mu).sum(axis=1) / norm
        vvar = 1.0 / norm

        assert mu.shape == (batch_size, q), (mu.shape, batch_size, q)
        # TODO: include self variance (Yvar) in 1 / sum(1/var)
        vvar = self._var_scale * vvar
        assert vvar.shape == (batch_size, q), (vvar.shape, batch_size, q)
        vvar = np.maximum(self._eps_var, vvar)

        return ENNNormal(mu, np.sqrt(vvar))
