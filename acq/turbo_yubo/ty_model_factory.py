import torch

from acq.fit_gp_turbo import train_gp as turbo_train_gp


def build_default_turbo_yubo_model(train_x: torch.Tensor, train_y: torch.Tensor):
    y = train_y
    if y.dim() > 1:
        y = y.squeeze(-1)
    mu = torch.median(y)
    sigma = y.std()
    if sigma < 1e-6:
        sigma = torch.tensor(1.0, dtype=y.dtype, device=y.device)
    y_std = (y - mu) / sigma

    gp = turbo_train_gp(train_x=train_x, train_y=y_std, use_ard=True, num_steps=50, hypers={})

    class _TurboPosteriorModel:
        def __init__(self, gp):
            self._gp = gp
            self.train_inputs = gp.train_inputs
            self.train_targets = gp.train_targets
            self.covar_module = gp.covar_module

        def posterior(self, X: torch.Tensor):
            with torch.no_grad():
                return self._gp.likelihood(self._gp(X))

    return _TurboPosteriorModel(gp)


def default_targeter(x_center: torch.Tensor, x_target: torch.Tensor):
    return x_target


class _NOOPPosterior:
    def __init__(self, X: torch.Tensor):
        self._X = X

    def sample(self, sample_shape: torch.Size):
        size = list(sample_shape)
        assert len(size) == 1, size
        size = [self._X.shape[0], size[0]]
        return torch.randn(size=size)


class TurboYUBONOOPModel:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self._train_y = train_y

    def posterior(self, X: torch.Tensor):
        return _NOOPPosterior(X)
