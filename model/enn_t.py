from dataclasses import dataclass
from typing import Optional

import faiss
import torch


@dataclass
class ENNNormalT:
    mu: torch.Tensor
    se: torch.Tensor

    def sample(self, num_samples, clip=None):
        if isinstance(num_samples, torch.Size):
            num_samples = list(num_samples)
            assert len(num_samples) == 1, num_samples
            num_samples = num_samples[0]
        size = list(self.se.shape)
        size.append(num_samples)
        eps = torch.randn(size, device=self.se.device, dtype=self.se.dtype)
        if clip is not None:
            eps = torch.clamp(eps, min=-clip, max=clip)
        return self.mu.unsqueeze(-1) + self.se.unsqueeze(-1) * eps


class EpistemicNearestNeighborsT:
    def __init__(self, k=3, small_world_M: Optional[int] = None, num_hnsw: Optional[int] = None):
        assert isinstance(k, int), k
        self.k = k
        self._num_dim: Optional[int] = None
        self._num_metrics: Optional[int] = None
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._train_y_var: Optional[torch.Tensor] = None
        self._eps_var: float = 1e-9
        self._var_scale: float = 1.0
        self._index = None
        self._small_world_M: Optional[int] = small_world_M
        self._hnsw_threshold: Optional[int] = num_hnsw

    def add(self, x: torch.Tensor, y: torch.Tensor, y_var: torch.Tensor):
        assert x.ndim == y.ndim == y_var.ndim == 2, (x.ndim, y.ndim, y_var.ndim)
        assert x.shape[0] == y.shape[0] == y_var.shape[0], (x.shape, y.shape, y_var.shape)
        if self._train_x is None:
            self._num_dim = x.shape[1]
            self._num_metrics = y.shape[1]
            self._train_x = x.new_empty((0, self._num_dim))
            self._train_y = y.new_empty((0, self._num_metrics))
            self._train_y_var = y_var.new_empty((0, self._num_metrics))
        assert x.shape[1] == self._num_dim, (x.shape[1], self._num_dim)
        assert y.shape[1] == self._num_metrics, (y.shape[1], self._num_metrics)
        assert y_var.shape[1] == self._num_metrics, (y_var.shape[1], self._num_metrics)
        assert self._train_x.device == x.device, (self._train_x.device, x.device)
        assert self._train_y.device == y.device, (self._train_y.device, y.device)
        assert self._train_y_var.device == y_var.device, (self._train_y_var.device, y_var.device)
        assert self._train_x.dtype == x.dtype, (self._train_x.dtype, x.dtype)
        assert self._train_y.dtype == y.dtype, (self._train_y.dtype, y.dtype)
        assert self._train_y_var.dtype == y_var.dtype, (self._train_y_var.dtype, y_var.dtype)
        if x.shape[0] > 0:
            self._train_x = torch.cat([self._train_x, x], dim=0)
            self._train_y = torch.cat([self._train_y, y], dim=0)
            self._train_y_var = torch.cat([self._train_y_var, y_var], dim=0)
            if self._small_world_M is None and self._hnsw_threshold is None:
                if x.is_cuda or x.dtype != torch.float32:
                    x_np = x.detach().cpu().to(dtype=torch.float32).numpy()
                else:
                    x_np = x.detach().numpy()
                if self._index is None:
                    self._index = faiss.IndexFlatL2(self._num_dim)
                self._index.add(x_np)
            else:
                total_n = self._train_x.shape[0]
                use_hnsw = False
                if self._small_world_M is not None:
                    if self._hnsw_threshold is None:
                        use_hnsw = True
                    else:
                        use_hnsw = total_n > self._hnsw_threshold
                needs_new_index = self._index is None
                needs_upgrade = use_hnsw and self._index is not None and not isinstance(self._index, faiss.IndexHNSWFlat)
                if needs_new_index or needs_upgrade:
                    x_all = self._train_x
                    if x_all.is_cuda or x_all.dtype != torch.float32:
                        x_all_np = x_all.detach().cpu().to(dtype=torch.float32).numpy()
                    else:
                        x_all_np = x_all.detach().numpy()
                    if use_hnsw:
                        index = faiss.IndexHNSWFlat(self._num_dim, int(self._small_world_M))
                    else:
                        index = faiss.IndexFlatL2(self._num_dim)
                    index.add(x_all_np)
                    self._index = index
                else:
                    if x.is_cuda or x.dtype != torch.float32:
                        x_np = x.detach().cpu().to(dtype=torch.float32).numpy()
                    else:
                        x_np = x.detach().numpy()
                    if self._index is None:
                        self._index = faiss.IndexFlatL2(self._num_dim)
                    self._index.add(x_np)

    def __len__(self) -> int:
        return 0 if self._train_x is None else self._train_x.shape[0]

    def posterior(self, x: torch.Tensor, *, k: Optional[int] = None, exclude_nearest: bool = False, observation_noise: bool = False) -> ENNNormalT:
        if k is None:
            k = self.k
        k = min(k, len(self))
        assert x.ndim == 2, x.shape
        b, d = x.shape
        if self._train_x is None or self._train_x.shape[0] == 0:
            mu = torch.zeros((b, self._num_metrics if self._num_metrics is not None else 0), device=x.device, dtype=x.dtype)
            se = torch.ones((b, self._num_metrics if self._num_metrics is not None else 0), device=x.device, dtype=x.dtype)
            return ENNNormalT(mu=mu, se=se)
        assert d == self._num_dim, (d, self._num_dim)
        if exclude_nearest:
            assert len(self) > 1, len(self)
        dist2s, idx = self._search(x.to(self._train_x.device, dtype=self._train_x.dtype), k=k, exclude_nearest=exclude_nearest)
        y = self._train_y[idx]
        y_var = self._train_y_var[idx]
        mu_t, se_t = self._calc_enn_normal(dist2s, y, y_var, observation_noise=observation_noise)
        mu_t = mu_t.to(device=x.device, dtype=x.dtype)
        se_t = se_t.to(device=x.device, dtype=x.dtype)
        return ENNNormalT(mu=mu_t, se=se_t)

    def set_var_scale(self, var_scale: float):
        self._var_scale = var_scale

    def _search(self, x: torch.Tensor, k: int, *, exclude_nearest: bool) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(self) > 0, len(self)
        if exclude_nearest:
            k = min(k + 1, len(self))
        assert self._index is not None
        k_eff = min(k, self._train_x.shape[0])
        if x.is_cuda or x.dtype != torch.float32:
            x_np = x.detach().cpu().to(dtype=torch.float32).numpy()
        else:
            x_np = x.detach().numpy()
        dist2_np, idx_np = self._index.search(x_np, k=k_eff)
        dist2 = torch.from_numpy(dist2_np).to(dtype=self._train_x.dtype, device=x.device)
        idx = torch.from_numpy(idx_np).to(dtype=torch.long, device=x.device)
        if exclude_nearest:
            dist2 = dist2[:, 1:]
            idx = idx[:, 1:]
        return dist2, idx

    def _calc_enn_normal(
        self, dist2s: torch.Tensor, y: torch.Tensor, y_var: torch.Tensor, observation_noise: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert dist2s.shape[:2] == y.shape[:2] == y_var.shape[:2], (dist2s.shape, y.shape, y_var.shape)
        batch_size, num_neighbors = dist2s.shape
        num_metrics = y.shape[-1]

        if num_neighbors == 1:
            mu = y[:, 0, :]
            vvar = torch.ones((batch_size, num_metrics), dtype=y.dtype, device=y.device)
            se = torch.sqrt(vvar)
            if not torch.isfinite(mu).all():
                mu = torch.zeros_like(mu)
            if not torch.isfinite(se).all() or (se <= 0).any():
                se = torch.ones_like(se)
            return mu, se

        dist2s_expanded = dist2s.unsqueeze(-1).expand(batch_size, num_neighbors, num_metrics)
        vvar = self._var_scale * dist2s_expanded + y_var
        w = 1.0 / (self._eps_var + vvar)
        norm = w.sum(dim=1)
        mu = (w * y).sum(dim=1) / norm
        vvar = 1.0 / norm
        vvar.clamp_min_(self._eps_var)
        se = torch.sqrt(vvar)
        if observation_noise:
            obS_noise_var = (w * y_var).sum(dim=1) / norm
            se = torch.sqrt(se**2 + obS_noise_var)

        if not torch.isfinite(mu).all():
            mu = torch.zeros_like(mu)
        if not torch.isfinite(se).all() or (se <= 0).any():
            se = torch.ones_like(se)

        return mu, se
