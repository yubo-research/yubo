import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qNoisyExpectedImprovement,
)
from botorch.optim import optimize_acqf
from botorch.utils import t_batch_mode_transform
from botorch.utils.probability.utils import ndtr as Phi
from botorch.utils.probability.utils import (
    phi,
)

# from botorch.sampling.normal import SobolQMCNormalSampler
# from IPython.core.debugger import set_trace
from torch.quasirandom import SobolEngine


class AcqITS(MCAcquisitionFunction):
    def __init__(self, model, num_X_samples=256, num_mcmc=10, ttype="msvar", sample_type=None, bounds=None, alt_ei=False, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._num_mcmc = num_mcmc
        self._num_X_samples = num_X_samples
        self.ttype = ttype
        self._alt_acqf = None

        X_0 = self.model.train_inputs[0].detach()
        self._num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)

        if sample_type == "mh" or ttype == "ei":
            self.X_max = self._find_max()
            self.Y_max = self.model.posterior(self.X_max).mean

        if self._num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
        else:
            if alt_ei:
                self._alt_acqf = qNoisyExpectedImprovement(self.model, X_baseline=self.model.train_inputs[0], prune_baseline=False)
            else:
                if sample_type == "2":
                    self.X_samples = self._sample_maxes_2(sobol_engine, num_X_samples)
                elif sample_type == "mh":
                    self.X_samples = self._sample_maxes_3(sobol_engine, num_X_samples, num_mcmc)
                elif sample_type == "sr":
                    self.X_samples = self._sample_maxes_sr(bounds, num_X_samples)
                elif sample_type == "sobol":
                    self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
                else:
                    self.X_samples = self._sample_maxes(sobol_engine, num_X_samples)

    def _find_max(self):
        X = self.model.train_inputs[0]
        num_dim = X.shape[-1]

        x_cand, _ = optimize_acqf(
            acq_function=PosteriorMean(self.model),
            bounds=torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return x_cand

    def _sample_maxes_3(self, sobol_engine, num_X_samples, num_mcmc):
        eps = 0.1
        X = torch.tile(self.X_max, (num_X_samples, 1))
        if False:
            X = sobol_engine.draw(num_X_samples // 2, dtype=self._dtype)
            X = torch.cat(
                (X, torch.tile(self.X_max, (num_X_samples // 2, 1))),
                dim=0,
            )

        for _ in range(num_mcmc):
            X_1 = X + eps * torch.randn(size=X.shape)
            Y = self.model.posterior(X).sample(torch.Size([1])).squeeze(-1)
            Y_1 = self.model.posterior(X_1).sample(torch.Size([1])).squeeze(-1)
            i = (X_1.min(dim=1).values >= 0) & (X_1.min(dim=1).values <= 1) & (Y_1 > Y).flatten()
            X[i] = X_1[i]
            # eps /= 3.1622776601683795
        return X

    def _sample_maxes_2(self, sobol_engine, num_X_samples):
        eps = 0.10
        X_0 = sobol_engine.draw(3 * num_X_samples, dtype=self._dtype)
        X_all = []
        for _ in range(3):
            X = X_0 + eps * torch.randn(size=X_0.shape)
            Y = self.model.posterior(X).sample(torch.Size([num_X_samples])).squeeze(-1)
            Y, i = torch.max(Y, dim=1)
            X_all.extend(X[i].unbind())
        X = torch.stack(X_all)
        i = torch.randint(len(X), (num_X_samples,))
        return X[i]

    def _sample_maxes(self, sobol_engine, num_X_samples):
        # X_obs = self.model.train_inputs[0]
        # Y_obs = self.model.posterior(X_obs).mean.squeeze(-1)
        # Y_max = Y_obs.max()
        # X_max = X_obs[Y_obs == Y_max]

        eps = 0.10
        X_samples = sobol_engine.draw(3 * num_X_samples, dtype=self._dtype)
        for _ in range(self._num_mcmc):
            X = None
            n_loop = 0
            while X is None or len(X) < num_X_samples:
                X_eps = X_samples + eps * torch.randn(size=X_samples.shape)
                X_eps = X_eps[(X_eps.min(dim=1).values > 0.0) & (X_eps.max(dim=1).values < 1.0)]
                if X is None:
                    X = X_eps
                else:
                    X = torch.cat((X, X_eps), dim=0)
                n_loop += 1
                assert n_loop < 10

            Y = self.model.posterior(X).sample(torch.Size([num_X_samples])).squeeze(-1)
            Y, i = torch.max(Y, dim=1)
            # doesn't help i = i[Y > Y_max]
            X_samples = X[i]
            eps /= 2
        i = torch.randint(len(X_samples), (num_X_samples,))
        return X_samples[i]

    @t_batch_mode_transform()
    def forward(self, X):
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        if self._alt_acqf:
            # for testing / comparison
            return self._alt_acqf(X)

        q = X.shape[-2]
        assert len(self.X_samples) >= 10 * q, "You should use num_X_samples >= 10*q"
        num_dim = X.shape[-1]
        num_obs = len(self.model.train_inputs[0])

        model_f = self.model.condition_on_observations(X=X, Y=self.model.posterior(X).mean)
        model_f.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + max(num_obs, q))) ** (1.0 / num_dim)

        mvn = model_f.posterior(self.X_samples)
        self.mvn = mvn

        # G-Optimality
        if self.ttype == "msvar":
            var_f = mvn.variance.squeeze()
            m = var_f.mean(dim=-1)
            s = var_f.std(dim=-1)
            return -(m + s)
        elif self.ttype == "mvar":
            var_f = mvn.variance.squeeze()
            m = var_f.mean(dim=-1)
            return -m
        elif self.ttype == "ei":
            mu_f = mvn.mean.squeeze()
            sd_f = mvn.stddev.squeeze()
            u = _scaled_improvement(mu_f, sd_f, self.Y_max)
            return -(sd_f * _ei_helper(u)).max(dim=-1).values
        elif self.ttype == "maxvar":
            var_f = mvn.variance.squeeze()
            return -var_f.max(dim=-1).values
        else:
            assert False, ("Unknown", self.ttype)


def _scaled_improvement(
    mean,
    sigma,
    best_f,
):
    return (mean - best_f) / sigma


def _ei_helper(u):
    return phi(u) + u * Phi(u)
