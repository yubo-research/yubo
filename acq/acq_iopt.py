import numpy as np
import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qSimpleRegret,
)
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIOpt(MCAcquisitionFunction):
    def __init__(self, model: Model, num_X_samples: int = 256, g_opt: bool = False, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

        self.g_opt = g_opt
        X_0 = self.model.train_inputs[0]
        self.num_dim = X_0.shape[-1]
        self.dtype = X_0.dtype

        sobol_engine = SobolEngine(self.num_dim, scramble=True)
        self.X_samples = sobol_engine.draw(num_X_samples, dtype=self.dtype)

        self._sr = qSimpleRegret(model)

        if len(X_0) == 0:
            self.p_iopt = 1.0
        else:
            self.p_iopt = min(1.0, self._mean_variance())

        self._num_sr = None
        # print ("PIOPT:", self.p_iopt)

    def _mean_variance(self):
        Y = self.model.posterior(self.X_samples)
        return Y.variance.mean().item()

    def _get_num_sr(self, X):
        if self._num_sr is None:
            q = X.shape[-2]
            self._num_sr = np.random.choice([0, 1], p=[self.p_iopt, 1 - self.p_iopt], size=(q,)).sum()
            # print ("NUM_SR:", X.shape, self._num_sr)
        return self._num_sr

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        q = X.shape[-2]
        q_sr = self._get_num_sr(X)
        q_iopt = q - q_sr
        assert q_iopt >= 0, (q, q_sr, q_iopt)
        assert q_sr >= 0, (q, q_sr, q_iopt)

        num_obs = len(self.model.train_inputs[0])

        if q_sr > 0:
            af_sr = self._sr(X[:, -q_sr:, :])
        else:
            af_sr = 0

        if q_iopt > 0:
            X_iopt = X[:, :q_iopt, :]
            Y = self.model.posterior(X_iopt).mean  # b x q x 1
            model_f = self.model.condition_on_observations(X=X_iopt, Y=Y)
            # model_f.covar_module.base_kernel.lengthscale *= (max(1, num_obs) / (max(1, num_obs) + q_iopt)) ** (1/self.num_dim)
            kernel = model_f.covar_module
            # Some models use `ScaleKernel(base_kernel=...)` while others use a bare kernel.
            if hasattr(kernel, "base_kernel"):
                kernel = kernel.base_kernel
            factor = ((1 + num_obs) / (1 + num_obs + q_iopt)) ** (1.0 / self.num_dim)
            # Adjusting kernel hyperparameters is part of the acquisition heuristic, not something
            # we want autograd to track.
            with torch.no_grad():
                kernel.lengthscale.mul_(factor)
            # var_f = model_f.posterior(self.X_samples, observation_noise=True).variance.squeeze()
            var_f = model_f.posterior(self.X_samples).variance.squeeze()

            if self.g_opt:
                af_iopt = -var_f.amax(dim=-1)  # nicer designs than mean, but much slower
            else:
                af_iopt = -var_f.mean(dim=-1)
        else:
            af_iopt = 0.0

        # print ("AF:", self.p_iopt, q_sr, X.shape, af_sr, q_iopt, af_iopt)
        return af_sr + af_iopt
