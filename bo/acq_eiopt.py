import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.quasirandom import SobolEngine

# from IPython.core.debugger import set_trace


# TODO: (higher-precision integral) optional adapted X samples
# TODO: optional MC joint samples of Y in fantasy model
#        to capture covariance/cross terms
class AcqEIOpt(MCAcquisitionFunction):
    def __init__(self, model: Model, num_X_samples: int = 256, num_ts_models: int = None, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype

        sobol_engine = SobolEngine(num_dim, scramble=True)
        self.X_samples = sobol_engine.draw(num_X_samples, dtype=dtype)
        if num_ts_models:
            self.weights = self._calc_weights(model, self.X_samples, num_ts_models)
        else:
            self.weights = torch.ones(size=(len(self.X_samples),))
            self.weights = self.weights / self.weights.sum()

    def _calc_weights(self, model, X_samples, num_ts_models):
        gp_ts = self._thompson_sample_models(model, num_ts_models)
        y = gp_ts.posterior(X_samples).sample().squeeze(0).squeeze(-1).detach()
        i_best = torch.argmax(y, dim=-1)
        i, counts = torch.unique(i_best, return_counts=True)
        p_best = torch.zeros(size=(len(X_samples),)).type(y.dtype)
        p_best[i] = counts.type(y.dtype)
        return p_best / p_best.sum()

    def _thompson_sample_models(self, model, num_ts_models):
        x = model.train_inputs[0].detach()
        x = x.repeat(num_ts_models, 1, 1)
        y = model.posterior(x).sample().squeeze(0).detach()

        model_ts = SingleTaskGP(x, y, model.likelihood)
        model_ts.initialize(**dict(model.named_parameters()))
        model_ts.eval()
        return model_ts

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        Y = self.model.posterior(X).mean  # b x q x 1
        model_t = self.model.condition_on_observations(X=X, Y=Y)

        var_t = model_t.posterior(self.X_samples, observation_noise=True).variance.squeeze()
        # set_trace()
        mean_var_t = (self.weights * var_t).sum(dim=-1)  # mean over X_samples
        # .mean(dim=-1)  # mean over X_samples

        return -mean_var_t
