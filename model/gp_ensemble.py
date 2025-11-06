import faiss
import numpy as np

from model.enn import ENNNormal
from model.k_randomized import KRandomized


class GPEnsemble:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, num_gps: int = 1):
        assert isinstance(train_x, np.ndarray)
        assert isinstance(train_y, np.ndarray)
        assert train_x.ndim == 2 and train_y.ndim == 2
        assert len(train_x) == len(train_y)
        assert isinstance(num_gps, int) and num_gps >= 1
        self._train_x = np.asarray(train_x, dtype=np.float64)
        self._train_y = np.asarray(train_y, dtype=np.float64)
        self._k_rands = [KRandomized(self._train_x) for _ in range(num_gps)]
        n, d = self._train_x.shape
        self._faiss_index = faiss.IndexFlatL2(d)
        self._train_x32 = self._train_x.astype(np.float32, copy=False)
        self._faiss_index.add(self._train_x32)

    def __call__(self, X):
        return self.posterior(X)

    def posterior(self, x, *, k=None, exclude_nearest=False):
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2
        assert isinstance(k, int) and k >= 1
        assert isinstance(exclude_nearest, bool)

        n, d = self._train_x.shape
        b = x.shape[0]
        assert x.shape[1] == d

        mu = np.zeros((b, self._train_y.shape[1]))
        se = np.zeros((b, self._train_y.shape[1]))

        x32 = x.astype(np.float32, copy=False)
        k_eff = min(k, n)
        dists2s, idxs = self._faiss_index.search(x32, k_eff)

        for t in range(b):
            xt = x[t]
            nn_idx = idxs[t]
            if exclude_nearest:
                assert len(nn_idx) > 1, "exclude_nearest requires k > 1"
                idx = nn_idx[1:]
            else:
                idx = nn_idx

            if not exclude_nearest and float(dists2s[t, 0]) == 0.0:
                mu[t, :] = self._train_y[nn_idx[0]]
                se[t, :] = 0.0
                continue

            Y_nn = self._train_y[idx]

            mu_members = []
            var_members = []
            for kr in self._k_rands:
                K_nn, k_star = kr.sub_k(idx, xt)
                alpha = np.linalg.solve(K_nn, Y_nn)
                mu_i = k_star @ alpha
                v = np.linalg.solve(K_nn, k_star)
                var_i = 1.0 - float(k_star @ v)
                var_i = max(0.0, var_i)
                mu_members.append(mu_i)
                var_members.append(var_i)

            if any(vi == 0.0 for vi in var_members):
                i0 = next(i for i, vi in enumerate(var_members) if vi == 0.0)
                mu[t, :] = mu_members[i0]
                se[t, :] = 0.0
            else:
                precisions = np.array([1.0 / vi for vi in var_members], dtype=np.float64)
                mu_stack = np.stack(mu_members, axis=0)
                weighted_mu = (precisions[:, None] * mu_stack).sum(axis=0)
                tau = precisions.sum()
                mu_comb = weighted_mu / tau
                var_comb = 1.0 / tau
                mu[t, :] = mu_comb
                se[t, :] = np.sqrt(var_comb)

        return ENNNormal(mu, se)
