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


class EpistemicNearestNeighbors:
    # TODO: train_YVar; treat as third metric in acquisition function b/c not calibrate to epistemic var
    def __init__(self, k=3, small_world_M=None):
        self.k = k
        self._num_dim = None
        self._num_metrics = None
        self._train_x = None
        self._train_y = None
        self._eps_var = 1e-9
        self._var_scale = 1.0
        self._lookup = None
        self._index = None
        self._small_world_M = small_world_M

    def add(self, x, y):
        assert x.ndim == y.ndim == 2, (x.ndim, y.ndim)
        assert len(x) == len(y), (len(x), len(y))
        if self._train_x is None:
            self._num_dim = x.shape[1]
            self._num_metrics = y.shape[1]
            self._train_x = np.empty((0, self._num_dim))
            self._train_y = np.empty((0, self._num_metrics))
            if self._small_world_M is not None:
                self._index = faiss.IndexHNSW(self._num_dim, M=self._small_world_M)
            else:
                self._index = faiss.IndexFlatL2(self._num_dim)
        self._index.add(x)
        self._train_x = np.append(self._train_x, x, axis=0)
        self._train_y = np.append(self._train_y, y, axis=0)
        self._num_obs = self._train_x.shape[0]
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
        i = np.where(dist > 1e-4)[0]
        if len(i) > 0:
            print(f"WARN: {len(i)} points may not be in training data, max(dist) = {dist.max()}")
        return idx.flatten()

    def about_neighbors(self, x, k=None):
        if k is None:
            k = self.k

        if self._train_x is None or len(self._train_x) == 0:
            return np.empty(shape=(0,), dtype=np.int64), np.empty(shape=(0,), dtype=np.float64)

        dist2s, idx = self._search(x, k=k)
        return idx, np.sqrt(dist2s)

    def neighbors(self, x, k=None):
        idx, _ = self.about_neighbors(x, k)
        if self._train_x is None:
            return np.empty((0, x.shape[1]))
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
                np.sqrt(vvar.squeeze(0)),
            )

        if exclude_nearest:
            dist2s, idx = self._search(x, k=k + 1)
            dist2s = dist2s[:, 1:]
            idx = idx[:, 1:]
        else:
            dist2s, idx = self._search(x, k=k)

        return self._calc_enn_normal(b, dist2s, idx, k)

    def _search(self, x, k):
        # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances?utm_source=chatgpt.com
        # "Faiss reports squared Euclidean (L2) distance..."
        dist2s, idx = self._index.search(x, k=k)
        return dist2s, idx

    def _calc_enn_normal(self, batch_size, dist2s, idx, k):
        q = 1

        mu = self._train_y[idx]
        assert mu.shape == (batch_size, k, self._num_metrics), (mu.shape, batch_size, k, self._num_metrics)
        vvar = np.expand_dims(dist2s, axis=-1)
        vvar = np.tile(vvar, (1, 1, self._num_metrics))
        assert vvar.shape == (batch_size, k, self._num_metrics), (vvar.shape, batch_size, k, self._num_metrics)

        w = 1.0 / (self._eps_var + vvar)
        assert np.all(np.isfinite(w)), w
        norm = w.sum(axis=1)
        # sum over k neighbors
        mu = (w * mu).sum(axis=1) / norm
        vvar = 1.0 / norm

        assert mu.shape == (batch_size, self._num_metrics), (mu.shape, batch_size, self._num_metrics)
        vvar = self._var_scale * vvar
        assert vvar.shape == (batch_size, self._num_metrics), (vvar.shape, batch_size, self._num_metrics)
        vvar = np.maximum(self._eps_var, vvar)

        return ENNNormal(mu, np.sqrt(vvar))
