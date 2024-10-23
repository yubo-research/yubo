import numpy as np
import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.optim import optimize_acqf
from botorch.utils import t_batch_mode_transform
from torch.quasirandom import SobolEngine

from sampling.pstar_sampler import PStarSampler
from sampling.stagger_thompson_sampler import StaggerThompsonSampler

from .acq_util import find_max


class AcqMTV(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        num_X_samples,
        ts_only=False,
        k_mcmc=100,
        sample_type="sts",
        num_refinements=30,
        x_max_type="find",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        print(
            f"AcqMTV: num_X_samples={num_X_samples} ts_only={ts_only} k_mcmc={k_mcmc} num_refinements = {num_refinements} sample_type={sample_type} x_max_type = {x_max_type}"
        )
        self.ts_only = ts_only

        X_0 = self.model.train_inputs[0].detach()
        num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self.device = X_0.device
        self.dtype = X_0.dtype
        self.weights = None

        self._k_mcmc = k_mcmc
        self._num_refinements = num_refinements
        self._x_max_type = x_max_type

        sobol_engine = SobolEngine(self._num_dim, scramble=True)

        if num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self.dtype).to(self.device)
        else:
            # TODO: Write acq's for pss and sts and get rid of ts_only.
            if sample_type == "pss":
                self._set_x_max()
                pss = PStarSampler(k_mcmc, self.model, self.X_max)
                self.X_samples = pss(num_X_samples)
            elif sample_type == "sts":
                self._set_x_max()
                if not ts_only:
                    self.X_samples = self._stagger_thompson_sampler(num_X_samples)
                else:
                    self.X_samples = "sts"
            elif sample_type == "pts":
                self._set_x_max()
                assert not ts_only, "Use designer pts directly"
                self.X_samples = self._pathwise_ts(num_X_samples)

            elif sample_type == "sobol":
                self.X_samples = sobol_engine.draw(num_X_samples, dtype=self.dtype)
            else:
                assert False, f"Unknown sample type [{sample_type}]"

        if hasattr(self.X_samples, "min"):
            assert self.X_samples.min() >= 0, self.X_samples
            assert self.X_samples.max() <= 1, self.X_samples

        if ts_only:
            # print("Using draw()")
            self.draw = self._draw

    def _pathwise_ts(self, num_X_samples):
        X_ts, _ = optimize_acqf(
            acq_function=PathwiseThompsonSampling(self.model),
            bounds=self._bounds(),
            q=num_X_samples,
            # num_restarts=100,
            raw_samples=128,
            # options={"batch_limit": 10, "maxiter": 200},
            num_restarts=3,
            # options={"batch_limit": num_ic, "maxiter": 100},
            options={"maxiter": 100},
            # batch_initial_conditions=batch_initial_conditions,
        )
        return X_ts

    def _bounds(self):
        return torch.tensor(
            [[0.0] * self._num_dim, [1.0] * self._num_dim],
            device=self.device,
            dtype=self.dtype,
        )

    def _set_x_max(self):
        if self._x_max_type == "find":
            self.X_max = find_max(
                self.model,
                bounds=self._bounds(),
            )
        elif self._x_max_type == "ts_meas":
            Y_ts = self.model.posterior(self.model.train_inputs[0]).rsample(torch.Size([1])).squeeze()
            i = torch.where(Y_ts == Y_ts.max())[0]
            self.X_max = self.model.train_inputs[0][i]
        elif self._x_max_type == "meas":
            Y = self.model.train_targets[0]
            i = torch.where(Y == Y.max())[0]
            self.X_max = self.model.train_inputs[0][i]
        else:
            assert False, ("Unknown x_max_type", self._x_max_type)
        # print("X_MAX:", self.X_max.device)

    def _draw(self, num_arms):
        if self.X_samples == "sts":
            return self._stagger_thompson_sampler(num_arms)
        assert len(self.X_samples) >= num_arms, (len(self.X_samples), num_arms)
        i = np.arange(len(self.X_samples))
        i = np.random.choice(i, size=(int(num_arms)), replace=False)
        return self.X_samples[i]

    def _stagger_thompson_sampler(self, num_samples):
        sts = StaggerThompsonSampler(self.model, self.X_max, num_samples=num_samples)
        sts.refine(self._num_refinements)
        return sts.samples()

    # def _calc_p_max_from_Y(self, Y):
    #     is_best = torch.argmax(Y, dim=-1)
    #     idcs, counts = torch.unique(is_best, return_counts=True)
    #     p_max = torch.zeros(Y.shape[-1])
    #     p_max[idcs] = counts / Y.shape[0]
    #     return p_max

    # def _p_target(self, X, num_Y_samples):
    #     mvn = self.model.posterior(X)
    #     Y = mvn.sample(torch.Size([num_Y_samples])).squeeze()
    #     assert torch.all((X >= 0) & (X <= 1))
    #     return self._calc_p_max_from_Y(Y)

    @t_batch_mode_transform()
    def forward(self, X):
        self.to(device=X.device)

        q = X.shape[-2]
        assert len(self.X_samples) > q, "You should use num_X_samples > q"

        mvn_a = self.model.posterior(X)
        model_f = self.model.condition_on_observations(X=X, Y=mvn_a.mean)
        mvn_f = model_f.posterior(self.X_samples, observation_noise=True)

        # print("DEVICE:", X.device)
        # I-Optimality
        var_f = mvn_f.variance.squeeze()
        # if self.weights is not None:
        #     var_f = self.weights.unsqueeze(0) * var_f
        m = var_f.mean(dim=-1)
        return -m
