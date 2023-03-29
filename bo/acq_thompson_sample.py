import gpytorch
import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqThompsonSample(MCAcquisitionFunction):
    def __init__(self, model: Model, q: int, num_X_samples_0: int = 1024, num_X_samples_per_dim: int = 0, b_joint_sampling: bool = True, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        X_0 = model.train_inputs[0]
        num_dim = X_0.shape[-1]
        num_X_samples = num_X_samples_0 + num_X_samples_per_dim * num_dim

        self.mv = self._mean_variance(num_X_samples)
        if len(X_0) == 0:
            p_global = 1
        else:
            p_global = min(1.0, self.mv)
        self.num_global_arms = np.random.binomial(q, p_global)
        self.num_local_arms = q - self.num_global_arms
        self.X_cand = []
        for _ in range(self.num_local_arms):
            self.X_cand.append(self._sample_local())
        for _ in range(self.num_global_arms):
            # TODO: sample all global arms at once, conditioned on local arms
            self.X_cand.append(self._sample_global(num_X_samples, b_joint_sampling))

        self.X_cand = torch.cat(self.X_cand, axis=0).to(X_0.device).type(X_0.dtype)

    def _sample_local(self):
        return self._find_max(self._get_noisy_model())

    def _sample_global(self, num_X_samples, b_joint_sampling):
        X_samples = self._sobol_samples(num_X_samples)

        if True:
            Y = self._sample_y(X_samples, b_joint_sampling)
            i = np.random.choice(np.where(Y == Y.max())[0])
            # i = np.random.randint(num_X_samples)
        else:
            var = []
            for i, X in enumerate(X_samples):
                var.append(self._future_mean_variance(torch.atleast_2d(X), X_samples))
            var = np.array(var)
            i = np.random.choice(np.where(var == var.min())[0])

        return torch.atleast_2d(X_samples[i])

    def _mean_variance(self, num_X_samples):
        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        X = torch.rand(size=(num_X_samples, num_dim))
        Y = self.model.posterior(X)
        return Y.variance.mean().item()

    def _get_noisy_model(self):
        X = self.model.train_inputs[0].detach()
        if len(X) == 0:
            return self.model
        Y = self.model.posterior(X, observation_noise=True).sample().squeeze(0).detach()
        model_ts = SingleTaskGP(X, Y, self.model.likelihood)
        model_ts.initialize(**dict(self.model.named_parameters()))
        model_ts.eval()
        return model_ts

    def _find_max(self, model):
        X = model.train_inputs[0]
        num_dim = X.shape[-1]

        x_max, _ = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return x_max

    def _sobol_samples(self, num_X_samples):
        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(num_dim, scramble=True)
        return sobol_engine.draw(num_X_samples).to(X_0.device).type(X_0.dtype)

    def _sample_y(self, X, b_joint_sampling):
        pred = self.model.likelihood(self.model(X))
        if b_joint_sampling:
            with torch.no_grad():
                with gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
                    Y = pred.sample().squeeze(0).squeeze(-1).detach()
        else:
            Y = pred.mean + pred.stddev * (torch.randn(size=X.shape[:-1]))
        return Y

    def _future_mean_variance(self, X, X_samples):
        with torch.no_grad():
            Y = self.model.posterior(X).mean  # b x q x 1
            model_t = self.model.condition_on_observations(X=X, Y=Y)
            # TODO: joint sampling / MC
            var = model_t.posterior(X_samples, observation_noise=True).variance.squeeze().mean(dim=-1)  # mean over X_samples
            return var.item()

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        assert False, "Just get X_cand"
