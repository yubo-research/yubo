import numpy as np
import torch

from model.enn import EpistemicNearestNeighbors
from model.enn_weighter import ENNWeighter


def _to_numpy(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _to_torch(a: np.ndarray, *, like: torch.Tensor):
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)


class ENNNormalWrapper:
    def __init__(self, X_like: torch.Tensor, mvn):
        self._X_like = X_like
        self._mvn = mvn

    @property
    def mu(self):
        return self._mvn.mu

    @property
    def se(self):
        return self._mvn.se

    def sample(self, sample_shape: torch.Size):
        samples_np = self._mvn.sample(sample_shape)
        samples_t = _to_torch(samples_np, like=self._X_like)
        return samples_t.permute(2, 0, 1).contiguous()


def build_turbo_yubo_enn_model(*, train_x: torch.Tensor, train_y: torch.Tensor, k: int = 3, small_world_M: int | None = None, weighting: str | None = None):
    x_t = train_x
    y_t = train_y
    if y_t.dim() > 1:
        y_t = y_t.squeeze(-1)

    x_np = _to_numpy(x_t)
    y_np = _to_numpy(y_t)[..., None]

    if weighting is None:
        enn_core = EpistemicNearestNeighbors(k=k, small_world_M=small_world_M)
    else:
        enn_core = ENNWeighter(k=k, small_world_M=small_world_M, weighting=weighting)

    enn_core.add(x_np, y_np)

    class _FakeKernel:
        def __init__(self, enn_core_ref, num_dim):
            self._enn_core_ref = enn_core_ref
            self._num_dim = num_dim

        @property
        def lengthscale(self):
            if hasattr(self._enn_core_ref, "weights") and self._enn_core_ref.weights is not None:
                weights_np = self._enn_core_ref.weights
            else:
                weights_np = np.ones(self._num_dim)
            weights_t = torch.from_numpy(weights_np).to(dtype=torch.float32)
            ls = (1.0 / weights_t).unsqueeze(0)
            return ls

    class _ENNModel:
        def __init__(self, x_like: torch.Tensor, y_like: torch.Tensor):
            self.train_inputs = (x_like.detach(),)
            self.train_targets = y_like.detach()
            if hasattr(enn_core, "_weights"):
                self.covar_module = _FakeKernel(enn_core, x_like.shape[1])
            else:
                self.covar_module = None
            if hasattr(enn_core, "set_x_center"):
                self.set_x_center = enn_core.set_x_center

        def posterior(self, X: torch.Tensor):
            X_np = _to_numpy(X)
            mvn = enn_core.posterior(X_np)

            return ENNNormalWrapper(X, mvn)

    return _ENNModel(x_t, y_t)
