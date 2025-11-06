import faiss
import numpy as np
import torch

from model.enn import ENNNormal
from model.k_randomized import KRandomized


class GPEnsembleT:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, num_gps: int = 1) -> None:
        assert isinstance(train_x, torch.Tensor)
        assert isinstance(train_y, torch.Tensor)
        assert train_x.ndim == 2 and train_y.ndim == 2, (train_x.ndim, train_y.ndim)
        assert train_x.shape[0] == train_y.shape[0], (train_x.shape, train_y.shape)
        assert isinstance(num_gps, int) and num_gps >= 1
        self._train_x = train_x.detach().cpu().to(dtype=torch.float64)
        self._train_y = train_y.detach().cpu().to(dtype=torch.float64)
        self._k_rands = [KRandomized(self._train_x.numpy()) for _ in range(num_gps)]
        n, d = self._train_x.shape
        self._faiss_index = faiss.IndexFlatL2(int(d))
        self._train_x32_np = self._train_x.to(dtype=torch.float32).numpy()
        self._faiss_index.add(self._train_x32_np)

        self._num_dim = int(d)
        self._num_metrics = int(self._train_y.shape[1])

    def __call__(self, X: torch.Tensor) -> ENNNormal:
        return self.posterior(X)

    def posterior(self, x: torch.Tensor, *, k: int | None = None, exclude_nearest: bool = False) -> ENNNormal:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 2
        assert isinstance(k, int) and k >= 1
        assert isinstance(exclude_nearest, bool)

        n = int(self._train_x.shape[0])
        b = int(x.shape[0])
        assert int(x.shape[1]) == self._num_dim

        mu = torch.zeros((b, self._num_metrics), dtype=torch.float64)
        se = torch.zeros((b, self._num_metrics), dtype=torch.float64)

        x32 = x.detach().cpu().to(dtype=torch.float32).numpy()
        k_eff = min(int(k), n)
        dists2s, idxs = self._faiss_index.search(x32, k_eff)

        for t in range(b):
            xt = x[t].detach().cpu().to(dtype=torch.float64).numpy()
            nn_idx = idxs[t]
            if exclude_nearest:
                assert len(nn_idx) > 1
                idx = nn_idx[1:]
            else:
                idx = nn_idx

            if (not exclude_nearest) and float(dists2s[t, 0]) == 0.0:
                mu[t, :] = self._train_y[int(nn_idx[0])]
                se[t, :] = 0.0
                continue

            Y_nn = self._train_y[idx]

            mu_members: list[torch.Tensor] = []
            var_members: list[float] = []
            for kr in self._k_rands:
                K_nn_np, k_star_np = kr.sub_k(np.asarray(idx, dtype=int), xt)
                K_nn = torch.from_numpy(K_nn_np).to(dtype=torch.float64)
                k_star = torch.from_numpy(k_star_np).to(dtype=torch.float64)
                alpha = torch.linalg.solve(K_nn, Y_nn)
                mu_i = k_star @ alpha
                v = torch.linalg.solve(K_nn, k_star)
                var_i = 1.0 - float(k_star @ v)
                if var_i < 0.0:
                    var_i = 0.0
                mu_members.append(mu_i)
                var_members.append(var_i)

            if any(vi == 0.0 for vi in var_members):
                i0 = next(i for i, vi in enumerate(var_members) if vi == 0.0)
                mu[t, :] = mu_members[i0]
                se[t, :] = 0.0
            else:
                precisions = torch.tensor([1.0 / vi for vi in var_members], dtype=torch.float64)
                mu_stack = torch.stack(mu_members, dim=0)
                weighted_mu = (precisions[:, None] * mu_stack).sum(dim=0)
                tau = precisions.sum()
                mu_comb = weighted_mu / tau
                var_comb = 1.0 / float(tau)
                mu[t, :] = mu_comb
                se[t, :] = torch.sqrt(torch.tensor(var_comb, dtype=torch.float64)).expand_as(mu_comb)

        mu_np = mu.numpy()
        se_np = se.numpy()
        return ENNNormal(mu_np, se_np)
