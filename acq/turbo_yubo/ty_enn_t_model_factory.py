import torch

from model.enn_t import EpistemicNearestNeighborsT
from model.enn_weighter_t import ENNWeighterT


class ENNNormalTWrapper:
    def __init__(self, mvn):
        self._mvn = mvn

    @property
    def mu(self):
        return self._mvn.mu

    @property
    def se(self):
        return self._mvn.se

    def sample(self, sample_shape: torch.Size):
        samples = self._mvn.sample(sample_shape)
        return samples.permute(2, 0, 1).contiguous()


def build_turbo_yubo_ennt_model(*, train_x: torch.Tensor, train_y: torch.Tensor, k: int = 3, weighting: str | None = None):
    x_t = train_x
    y_t = train_y
    if y_t.dim() > 1:
        y_t = y_t.squeeze(-1)
    y_t = y_t[..., None]
    y_var_t = torch.zeros_like(y_t)

    if weighting is None:
        enn_core = EpistemicNearestNeighborsT(k=k)
        enn_core.add(x_t, y_t, y_var_t)
    else:
        enn_core = ENNWeighterT(weighting=weighting, k=k)
        enn_core.add(x_t, y_t)

    class _ENNModelT:
        def __init__(self, x_like: torch.Tensor, y_like: torch.Tensor):
            self.train_inputs = (x_like.detach(),)
            self.train_targets = y_like.detach()
            if hasattr(enn_core, "weights"):

                class _FakeKernel:
                    def __init__(self, enn_core_ref, num_dim):
                        self._enn_core_ref = enn_core_ref
                        self._num_dim = num_dim

                    @property
                    def lengthscale(self):
                        try:
                            weights_t = self._enn_core_ref.weights
                        except Exception:
                            weights_t = torch.ones(self._num_dim, dtype=torch.float32)
                        ls = (1.0 / weights_t.to(torch.float32)).unsqueeze(0)
                        return ls

                self.covar_module = _FakeKernel(enn_core, x_like.shape[1])
            else:
                self.covar_module = None

        def posterior(self, X: torch.Tensor):
            mvn = enn_core.posterior(X)
            return ENNNormalTWrapper(mvn)

    return _ENNModelT(x_t, y_t)
