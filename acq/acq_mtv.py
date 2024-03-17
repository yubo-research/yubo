import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.optim import optimize_acqf
from botorch.utils import t_batch_mode_transform
from torch.quasirandom import SobolEngine

from sampling.pstar_sampler import PStarSampler


class AcqMTV(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        num_X_samples,
        ts_only=False,
        num_mcmc=100,
        sample_type="hnr",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        print(f"AcqMTV: num_X_samples={num_X_samples} ts_only={ts_only} num_mcmc={num_mcmc} sample_type={sample_type}")
        self.ts_only = ts_only

        X_0 = self.model.train_inputs[0].detach()
        num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self.device = X_0.device
        self.dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)

        if num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self.dtype)
            self.Y_max = 0.0
            self.Y_best = 0.0
        else:
            self.X_max = self._find_max()
            self.Y_max = self.model.posterior(self.X_max).mean
            if len(self.model.train_targets) > 0:
                i = torch.argmax(self.model.train_targets)
                self.Y_best = self.model.posterior(self.model.train_inputs[0][i][:, None].T).mean
            else:
                self.Y_best = self.Y_max

            if sample_type == "hnr":
                pss = PStarSampler(num_mcmc, self.model, self.X_max)
                self.X_samples = pss(num_X_samples)
            elif sample_type == "sobol":
                self.X_samples = sobol_engine.draw(num_X_samples, dtype=self.dtype)
            else:
                assert False, f"Unknown sample type [{sample_type}]"

        assert self.X_samples.min() >= 0, self.X_samples
        assert self.X_samples.max() <= 1, self.X_samples

        if ts_only:
            print("Using draw()")
            self.draw = self._draw

    def _draw(self, num_arms):
        assert len(self.X_samples) >= num_arms, (len(self.X_samples), num_arms)
        i = np.arange(len(self.X_samples))
        i = np.random.choice(i, size=(int(num_arms)), replace=False)
        return self.X_samples[i]

    def _find_max(self):
        X = self.model.train_inputs[0]

        x_cand, _ = optimize_acqf(
            acq_function=PosteriorMean(self.model),
            bounds=torch.tensor([[0.0] * self._num_dim, [1.0] * self._num_dim], device=self.device, dtype=self.dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )

        Y_cand = self.model.posterior(x_cand).mean
        if len(self.model.train_targets) > 0:
            i = torch.argmax(self.model.train_targets)
            Y_tgt = self.model.posterior(self.model.train_inputs[0][i][:, None].T).mean
            if Y_tgt > Y_cand:
                x_cand = self.model.train_inputs[0][i, :][:, None].T

        return x_cand

    @t_batch_mode_transform()
    def forward(self, X):
        self.to(device=X.device)

        q = X.shape[-2]
        assert len(self.X_samples) > q, "You should use num_X_samples > q"

        mvn_a = self.model.posterior(X)
        model_f = self.model.condition_on_observations(X=X, Y=mvn_a.mean)
        mvn_f = model_f.posterior(self.X_samples, observation_noise=True)

        # I-Optimality
        var_f = mvn_f.variance.squeeze()
        m = var_f.mean(dim=-1)
        return -m
