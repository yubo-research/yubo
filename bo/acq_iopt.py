import numpy as np
import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qNoisyExpectedImprovement,
)
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIOpt(MCAcquisitionFunction):
    def __init__(
        self, model: Model, num_X_samples: int = 256, num_p_samples: int = 256, use_sqrt: bool = False, explore_only=False, prune_baseline=True, **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)

        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(num_dim, scramble=True)
        X_samples = sobol_engine.draw(num_X_samples, dtype=X_0.dtype)
        self.register_buffer("X_samples", X_samples)

        p_iopt = self._integrated_variance(model, num_dim, num_X_samples)
        if use_sqrt:
            p_iopt = np.sqrt(p_iopt)
        if explore_only or np.random.uniform() < p_iopt:
            print("IOPT")
            self.acqf = None
        else:
            self.acqf = qNoisyExpectedImprovement(model, X_baseline=X_0, prune_baseline=prune_baseline)

    def _integrated_variance(self, model, num_dim, num_X_samples):
        X = torch.rand(size=(num_X_samples, num_dim))
        Y = model.posterior(X)
        return Y.variance.mean().item()

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        if self.acqf is not None:
            return self.acqf(X)

        Y = self.model.posterior(X).mean  # b x q x 1
        model_t = self.model.condition_on_observations(X=X, Y=Y)

        var_t = model_t.posterior(self.X_samples, observation_noise=True).variance.squeeze().mean(dim=-1)  # mean over X_samples

        return -var_t
