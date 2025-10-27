import faiss
import numpy as np

from model.enn import ENNNormal


class KernlizedEpistemicNearestNeighbors:
    # TODO: Try linear regression for weights
    def __init__(self, lengthscales: np.ndarray, k=3, small_world_M=None):
        assert len(lengthscales.shape) == 1, lengthscales.shape
        self.lengthscales = lengthscales
        self.k = k
        self._num_dim = None
        self._num_metrics = None
        self._train_x = None
        self._train_y = None
        self._index = None
        self._small_world_M = small_world_M
        self._eps_var = 1e-9
        self._var_scale = 1.0

    def add(self, x, y):
        assert x.ndim == y.ndim == 2, (x.ndim, y.ndim)
        assert len(x) == len(y), (len(x), len(y))
        if self._train_x is None:
            self._num_dim = x.shape[1]
            self._num_metrics = y.shape[1]
            assert self._num_dim == len(self.lengthscales), (self._num_dim, len(self.lengthscales))
            self._train_x = np.empty((0, self._num_dim))
            self._train_y = np.empty((0, self._num_metrics))
            if self._small_world_M is not None:
                base_index = faiss.IndexHNSWFlat(self._num_dim, self._small_world_M)
            else:
                base_index = faiss.IndexFlatL2(self._num_dim)
            self._index = base_index
        self._index.add(x.astype(np.float32))
        self._train_x = np.append(self._train_x, x, axis=0)
        self._train_y = np.append(self._train_y, y, axis=0)

    def __len__(self):
        return 0 if self._train_x is None else len(self._train_x)

    def __call__(self, X):
        return self.posterior(X)

    def posterior(self, x, *, k=None):
        if k is None:
            k = self.k
        k = min(k, len(self))

        x = np.array(x)
        assert len(x.shape) == 2, ("NYI: Joint sampling", x.shape)
        b, d = x.shape

        if self._num_dim is None:
            assert d == len(self.lengthscales), (d, len(self.lengthscales))
            self._num_dim = d
            self._num_metrics = 1

        assert d == self._num_dim, (d, self._num_dim)

        if len(self) == 0:
            mu = np.zeros((b, self._num_metrics))
            vvar = np.ones((b, self._num_metrics))
            return ENNNormal(mu, np.sqrt(vvar))

        k = min(k, len(self))
        dist2s, idx = self._index.search(x.astype(np.float32), k=k)

        return self._calc_kenn_normal(x, idx)

    def _calc_kenn_normal(self, x, idx):
        batch_size, num_neighbors = idx.shape
        y = self._train_y[idx]

        x_rep = np.repeat(x[:, None, :], num_neighbors, axis=1)
        x_train = self._train_x[idx]

        diff = x_rep - x_train
        scaled_diff = diff / self.lengthscales
        dist_scaled_sq = np.sum(scaled_diff**2, axis=-1)

        k_vals = np.exp(-0.5 * dist_scaled_sq)
        k_vals = np.maximum(k_vals, self._eps_var)

        k_vals = np.expand_dims(k_vals, axis=-1)
        k_vals = np.tile(k_vals, (1, 1, self._num_metrics))

        mu = (k_vals * y).sum(axis=1) / k_vals.sum(axis=1)

        k_sum = k_vals.sum(axis=1)
        vvar = 1.0 / k_sum
        vvar = self._var_scale * vvar
        vvar = np.maximum(self._eps_var, vvar)

        assert mu.shape == (batch_size, self._num_metrics), (mu.shape, batch_size, self._num_metrics)
        assert vvar.shape == (batch_size, self._num_metrics), (vvar.shape, batch_size, self._num_metrics)

        return ENNNormal(mu, np.sqrt(vvar))
