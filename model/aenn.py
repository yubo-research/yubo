import faiss
import numpy as np

from model.enn import ENNNormal
from sampling.nncd import nncd_weights


class AdditiveEpistemicNearestNeighbors:
    def __init__(self, k=3):
        assert isinstance(k, int), k
        assert k >= 1, k
        self.k = k
        self._num_dim = None
        self._num_metrics = None
        self._train_x = None
        self._train_y = None
        self._eps_var = 1e-9
        self._var_scale = 1.0
        self._weights = None
        self._index = None

    @property
    def weights(self):
        return self._weights

    def add(self, x, y):
        assert x.ndim == y.ndim == 2, (x.ndim, y.ndim)
        assert len(x) == len(y), (len(x), len(y))
        if self._train_x is None:
            self._num_dim = x.shape[1]
            self._num_metrics = y.shape[1]
            self._train_x = np.empty((0, self._num_dim))
            self._train_y = np.empty((0, self._num_metrics))
            self._index = faiss.IndexFlatL2(self._num_dim)
        self._index.add(x.astype(np.float32))
        self._train_x = np.append(self._train_x, x, axis=0)
        self._train_y = np.append(self._train_y, y, axis=0)
        self._fit_beta()

    def __len__(self):
        return 0 if self._train_x is None else len(self._train_x)

    def __call__(self, x):
        return self.posterior(x)

    def posterior(self, x):
        x = np.array(x)
        assert len(x.shape) == 2, ("NYI: Joint sampling", x.shape)
        b, d = x.shape

        if self._train_x is None or self._train_x.shape[0] == 0:
            mu = np.zeros((b, 1))
            se = np.ones((b, 1))
            return ENNNormal(mu, se)

        assert d == self._num_dim, (d, self._num_dim)

        if self._weights is None:
            if len(self) == 0:
                mu = np.zeros((b, 1))
                se = np.ones((b, 1))
                return ENNNormal(mu, se)
            k = min(self.k, len(self))
            _, idx = self._index.search(x.astype(np.float32), k=k)
            neighbors = self._train_x[idx]
            diffs = x[:, None, :] - neighbors
            dist2 = (diffs**2).sum(axis=-1)
            y_neighbors = self._train_y[idx]
            return self._calc_enn_normal(dist2, y_neighbors)

        k = min(self.k, len(self))
        _, idx = self._index.search(x.astype(np.float32), k=k)

        neighbors = self._train_x[idx]
        diffs_all = x[:, None, :] - neighbors
        dist2_all = diffs_all**2  # (b, k, D)

        y_neighbors = self._train_y[idx]  # (b, k, M)

        mu_d = np.zeros((b, self._num_dim, self._num_metrics))
        se_d = np.zeros((b, self._num_dim, self._num_metrics))

        for dim in range(self._num_dim):
            mvn = self._calc_enn_normal(dist2_all[:, :, dim], y_neighbors)
            mu_d[:, dim, :] = mvn.mu
            se_d[:, dim, :] = mvn.se

        mu = np.zeros((b, self._num_metrics))
        se = np.zeros((b, self._num_metrics))

        for m in range(self._num_metrics):
            mu[:, m] = np.sum(self._weights[:, m] * mu_d[:, :, m], axis=1)
            se[:, m] = np.sqrt(np.sum(self._weights[:, m] ** 2 * se_d[:, :, m] ** 2, axis=1))

        return ENNNormal(mu, se)

    def _fit_beta(self):
        n = len(self._train_x)

        if n < 2:
            self._weights = None
            return

        mu_d_matrix = np.zeros((n, self._num_dim, self._num_metrics))

        k_actual = min(self.k, n - 1)
        k_search = k_actual + 1
        _, idx_all = self._index.search(self._train_x.astype(np.float32), k=k_search)

        mask = idx_all != np.arange(n)[:, None]
        idx_filtered_all = np.where(mask, idx_all, -1)

        idx_neighbors = np.zeros((n, k_actual), dtype=np.int64)
        for i in range(n):
            row = idx_filtered_all[i]
            row = row[row != -1][:k_actual]
            if len(row) < k_actual:
                pad_val = row[-1] if len(row) > 0 else 0
                row = np.concatenate([row, np.full(k_actual - len(row), pad_val, dtype=np.int64)])
            idx_neighbors[i] = row

        neighbors_all = self._train_x[idx_neighbors]  # (n, k, D)
        diffs_all = self._train_x[:, None, :] - neighbors_all  # (n, k, D)
        dist2_all = diffs_all**2  # (n, k, D)
        y_neighbors_all = self._train_y[idx_neighbors]  # (n, k, M)

        for dim in range(self._num_dim):
            mvn = self._calc_enn_normal(dist2_all[:, :, dim], y_neighbors_all)
            mu_d_matrix[:, dim, :] = mvn.mu

        X = np.transpose(mu_d_matrix, (2, 0, 1))
        Y = self._train_y.T[..., None]
        W = nncd_weights(y=Y, x=X, iters_per_dimension=5, eps=1e-9)
        assert W.shape == (self._num_metrics, self._num_dim)
        self._weights = W.T
        assert np.all(self._weights >= 0), ("Negative beta coefficients found", self._weights)
        s = self._weights.sum(axis=0)
        assert np.allclose(s, 1.0, atol=1e-6), ("Beta columns must sum to 1", s)

    def _calc_enn_normal(self, dist2s, y):
        assert len(dist2s) == len(y), (len(dist2s), len(y))
        assert y.shape[-1] == self._num_metrics, (y.shape, self._num_metrics)

        batch_size, num_neighbors, _ = y.shape

        mu = y
        assert mu.shape == (batch_size, num_neighbors, self._num_metrics), (mu.shape, batch_size, num_neighbors, self._num_metrics)
        vvar = np.expand_dims(dist2s, axis=-1)
        vvar = np.tile(vvar, (1, 1, self._num_metrics))
        assert vvar.shape == (batch_size, num_neighbors, self._num_metrics), (vvar.shape, batch_size, num_neighbors, self._num_metrics)

        w = 1.0 / (self._eps_var + vvar)
        assert np.all(np.isfinite(w)), w
        norm = w.sum(axis=1)
        mu = (w * mu).sum(axis=1) / norm
        vvar = 1.0 / norm

        assert mu.shape == (batch_size, self._num_metrics), (mu.shape, batch_size, self._num_metrics)
        vvar = self._var_scale * vvar
        assert vvar.shape == (batch_size, self._num_metrics), (vvar.shape, batch_size, self._num_metrics)
        vvar = np.maximum(self._eps_var, vvar)

        return ENNNormal(mu, np.sqrt(vvar))
