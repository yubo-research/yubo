import numpy as np
import torch

from model.gp_ensemble_t import GPEnsembleT


def _to_numpy(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _to_torch(a: np.ndarray, *, like: torch.Tensor):
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)


class _ENNNormalWrapper:
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


def build_turbo_yubo_ge_model(
    *, train_x: torch.Tensor, train_y: torch.Tensor, k: int = 3, num_gps: int = 1
):
    x_t = train_x
    y_t = train_y
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(-1)

    core = GPEnsembleT(train_x=x_t, train_y=y_t, num_gps=num_gps)

    class _GPEModel:
        def __init__(self, x_like: torch.Tensor, y_like: torch.Tensor):
            self.train_inputs = (x_like.detach(),)
            self.train_targets = y_like.detach()
            self.covar_module = None

        def posterior(self, X: torch.Tensor):
            mvn = core.posterior(X, k=k)
            return _ENNNormalWrapper(X, mvn)

    return _GPEModel(x_t, y_t)
