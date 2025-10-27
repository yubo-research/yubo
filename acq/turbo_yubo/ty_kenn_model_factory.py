import numpy as np
import torch

from model.kenn import KernlizedEpistemicNearestNeighbors


def _to_numpy(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _to_torch(a: np.ndarray, *, like: torch.Tensor):
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)


def build_turbo_yubo_kenn_model(
    *, train_x: torch.Tensor, train_y: torch.Tensor, k: int = 3, small_world_M: int | None = None, lengthscales: np.ndarray | None = None
):
    x_t = train_x
    y_t = train_y
    if y_t.dim() > 1:
        y_t = y_t.squeeze(-1)

    x_np = _to_numpy(x_t)
    y_np = _to_numpy(y_t)[..., None]

    if lengthscales is None:
        lengthscales = np.ones(x_np.shape[1])

    kenn_core = KernlizedEpistemicNearestNeighbors(lengthscales=lengthscales, k=k, small_world_M=small_world_M)
    kenn_core.add(x_np, y_np)

    class _FakeKernel:
        def __init__(self, lengthscales_np):
            self._lengthscales_np = lengthscales_np

        @property
        def lengthscale(self):
            lengthscales_t = torch.from_numpy(self._lengthscales_np).to(dtype=torch.float32)
            return lengthscales_t.unsqueeze(0)

    class _KENNModel:
        def __init__(self, x_like: torch.Tensor, y_like: torch.Tensor):
            self.train_inputs = (x_like.detach(),)
            self.train_targets = y_like.detach()
            self.covar_module = _FakeKernel(lengthscales)

        def posterior(self, X: torch.Tensor):
            X_np = _to_numpy(X)
            mvn = kenn_core.posterior(X_np)

            class _P:
                def __init__(self, X_like: torch.Tensor, mvn):
                    self._X_like = X_like
                    self._mvn = mvn

                def sample(self, sample_shape: torch.Size):
                    samples_np = self._mvn.sample(sample_shape)
                    samples_t = _to_torch(samples_np, like=self._X_like)
                    return samples_t.permute(2, 0, 1).contiguous()

            return _P(X, mvn)

    return _KENNModel(x_t, y_t)
