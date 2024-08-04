import numpy as np
import torch
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.generation import MaxPosteriorSampling
from botorch.utils import t_batch_mode_transform
from torch.quasirandom import SobolEngine

from sampling.pstar_sampler import PStarSampler
from sampling.pstar_stagger import PStarStagger
from sampling.stagger import StaggerIS
from sampling.stagger_sobol import StaggerSobol

from .acq_util import find_max


class AcqMTV(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        num_X_samples,
        ts_only=False,
        k_mcmc=100,
        sample_type="hnr",
        # num_samples_per_dimension=10,
        # num_Y_samples=1024,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        print(f"AcqMTV: num_X_samples={num_X_samples} ts_only={ts_only} k_mcmc={k_mcmc} sample_type={sample_type}")
        self.ts_only = ts_only

        X_0 = self.model.train_inputs[0].detach()
        num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self.device = X_0.device
        self.dtype = X_0.dtype
        self.weights = None
        self.k_mcmc = k_mcmc

        sobol_engine = SobolEngine(self._num_dim, scramble=True)

        if num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self.dtype).to(self.device)
        else:
            if sample_type == "hnr":
                self._set_x_max()
                pss = PStarSampler(k_mcmc, self.model, self.X_max)
                self.X_samples = pss(num_X_samples)
            elif sample_type == "pss":
                self._set_x_max()
                if not ts_only:
                    self.X_samples = self._pstar_stagger(num_X_samples)
                else:
                    self.X_samples = "pss"
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

    def _set_x_max(self):
        self.X_max = find_max(
            self.model,
            bounds=torch.tensor(
                [[0.0] * self._num_dim, [1.0] * self._num_dim],
                device=self.device,
                dtype=self.dtype,
            ),
        )
        print("X_MAX:", self.X_max.device)

    def _draw(self, num_arms):
        if self.X_samples == "pss":
            return self._pstar_stagger(num_arms)
        assert len(self.X_samples) >= num_arms, (len(self.X_samples), num_arms)
        i = np.arange(len(self.X_samples))
        i = np.random.choice(i, size=(int(num_arms)), replace=False)
        return self.X_samples[i]

    def _pstar_stagger(self, num_samples):
        pss = PStarStagger(self.model, self.X_max, num_samples=num_samples)
        pss.refine(self.k_mcmc)
        return pss.samples()

    def _stagger_sobol(self, num_candidates, num_ts):
        ss = StaggerSobol(self.X_max)
        sampler = ss.get_sampler(num_proposal_points=num_candidates)
        X_unif = sampler.sample_uniform(num_samples=num_candidates)

        with torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_samples = thompson_sampling(X_unif, num_samples=num_ts)

        return X_samples

    def _calc_p_max_from_Y(self, Y):
        is_best = torch.argmax(Y, dim=-1)
        idcs, counts = torch.unique(is_best, return_counts=True)
        p_max = torch.zeros(Y.shape[-1])
        p_max[idcs] = counts / Y.shape[0]
        return p_max

    def _p_target(self, X, num_Y_samples):
        mvn = self.model.posterior(X)
        Y = mvn.sample(torch.Size([num_Y_samples])).squeeze()
        assert torch.all((X >= 0) & (X <= 1))
        return self._calc_p_max_from_Y(Y)

    def _stagger_is(self, num_samples_per_dimension, num_Y_samples, num_X_samples, conv_thresh=0.1):
        stagger = StaggerIS(self.X_max)
        X_and_p_target = None
        for i_iter in range(10):
            X = stagger.ask(
                num_samples_per_dimension=num_samples_per_dimension,
                X_and_p_target=X_and_p_target,
            )
            if stagger.convergence_criterion() < conv_thresh:
                break
            X_and_p_target = (X, self._p_target(X, num_Y_samples))

        # TODO: Maybe use more base samples
        sampler = stagger.sampler(num_base_samples=num_X_samples)
        X = sampler.ask()
        sampler.tell(self._p_target(X, num_Y_samples))
        return sampler.importance_sample(num_samples=num_X_samples)

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
