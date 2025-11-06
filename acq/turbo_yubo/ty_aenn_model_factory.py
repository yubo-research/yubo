import numpy as np
import torch

from acq.turbo_yubo.ty_enn_model_factory import ENNNormalWrapper
from model.aenn import AdditiveEpistemicNearestNeighbors


def _to_numpy(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _to_torch(a: np.ndarray, *, like: torch.Tensor):
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)


def build_turbo_yubo_aenn_model(*, train_x: torch.Tensor, train_y: torch.Tensor, k: int = 3):
    x_t = train_x
    y_t = train_y
    if y_t.dim() > 1:
        y_t = y_t.squeeze(-1)

    x_np = _to_numpy(x_t)
    y_np = _to_numpy(y_t)[..., None]

    aenn_core = AdditiveEpistemicNearestNeighbors(k=k)
    aenn_core.add(x_np, y_np)

    class _FakeKernel:
        def __init__(self, aenn_core_ref, num_dim):
            self._aenn_core_ref = aenn_core_ref
            self._num_dim = num_dim

        @property
        def lengthscale(self):
            weights_np = self._aenn_core_ref.weights
            if weights_np is None:
                weights_np = np.ones(self._num_dim)
            if isinstance(weights_np, np.ndarray) and weights_np.ndim == 2:
                if weights_np.shape[1] >= 1:
                    weights_np = weights_np[:, 0]
                else:
                    weights_np = np.ones(self._num_dim)
            weights_np = np.asarray(weights_np, dtype=np.float32)
            weights_np = np.where(weights_np <= 0, 1.0, weights_np)
            ls_np = 1.0 / weights_np
            ls_t = torch.from_numpy(ls_np.astype(np.float32))
            return ls_t.unsqueeze(0)

    class _AENNModel:
        def __init__(self, x_like: torch.Tensor, y_like: torch.Tensor):
            self.train_inputs = (x_like.detach(),)
            self.train_targets = y_like.detach()
            self.covar_module = _FakeKernel(aenn_core, x_like.shape[1])

        def posterior(self, X: torch.Tensor):
            X_np = _to_numpy(X)
            mvn = aenn_core.posterior(X_np)

            return ENNNormalWrapper(X, mvn)

    return _AENNModel(x_t, y_t)
