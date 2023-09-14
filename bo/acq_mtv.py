import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from botorch.utils.probability.utils import ndtr as Phi
from botorch.utils.probability.utils import (
    phi,
)

# from IPython.core.debugger import set_trace
from torch.quasirandom import SobolEngine


def acqf_pm(mvn, Y):
    return Y.mean(dim=0)


def acqf_ucb(mvn, Y):
    mu = Y.mean(dim=0)
    sg = Y.std(dim=0)
    return mu + sg


class AcqMTV(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        num_X_samples,
        ttype,
        beta=0,
        num_mcmc=5,
        num_Y_samples=1,
        beta_ucb=2,
        sample_type="mh",
        alt_acqf=None,
        acqf_pstar="pm",
        num_pstar_samples=1,
        lengthscale_correction=True,
        eps_0=0.1,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.num_mcmc = num_mcmc
        self.num_X_samples = num_X_samples
        self.ttype = ttype
        self.beta = beta
        self.beta_ucb = beta_ucb
        self._alt_acqf = alt_acqf
        self._k_eps = 0.5
        self._lengthscale_correction = lengthscale_correction
        self._eps_0 = eps_0
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

        X_0 = self.model.train_inputs[0].detach()
        self._num_obs = X_0.shape[0]
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype
        self._e_0 = torch.randn(size=(num_X_samples, 1), dtype=self._dtype)

        sobol_engine = SobolEngine(self._num_dim, scramble=True)

        if self._num_obs == 0:
            self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
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

            if sample_type == "mh":
                if acqf_pstar == "pm":
                    acqf_pstar = acqf_pm
                elif acqf_pstar == "ucb":
                    acqf_pstar = acqf_ucb
                else:
                    assert False, f"Unknown acqf_pstar = {acqf_pstar}"
                self.X_samples = self._sample_maxes_mh(acqf_pstar, sobol_engine, num_X_samples, num_mcmc, num_pstar_samples)
            elif sample_type == "sobol":
                self.X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
            else:
                assert False, f"Unknown sample type [{sample_type}]"

        assert self.X_samples.min() >= 0, self.X_samples
        assert self.X_samples.max() <= 1, self.X_samples

        if ttype == "ts":
            print("Using draw()")
            self.draw = self._draw

    def _draw(self, num_arms):
        assert len(self.X_samples) >= num_arms, (len(self.X_samples), num_arms)
        i = np.arange(len(self.X_samples))
        i = np.random.choice(i, size=(int(num_arms)), replace=False)
        return self.X_samples[i]

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

    def _sample_maxes_mh(self, acqf_pstar, sobol_engine, num_X_samples, num_mcmc, num_pstar_samples):
        X_max = self._find_max()
        eps = self._eps_0  # 0.1

        X = torch.tile(X_max, (num_X_samples, 1))

        if False:
            X = sobol_engine.draw(num_X_samples // 2, dtype=self._dtype)
            X = torch.cat(
                (X, torch.tile(X_max, (num_X_samples // 2, 1))),
                dim=0,
            )

        for _ in range(num_mcmc):
            X_1 = X + eps * torch.randn(size=X.shape)
            X_both = torch.cat((X, X_1), dim=0)
            mvn = self.model.posterior(X_both, observation_noise=True)
            Y_both = mvn.sample(torch.Size([num_pstar_samples])).squeeze(-1).squeeze(0)
            af_both = acqf_pstar(mvn, Y_both)
            af = af_both[: len(X)]
            af_1 = af_both[len(X) :]
            i = (X_1.min(dim=1).values >= 0) & (X_1.max(dim=1).values <= 1) & (af_1 > af).flatten()

            X[i] = X_1[i]
            eps = self._k_eps * eps
        return X

    @t_batch_mode_transform()
    def forward(self, X):
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        if self._num_obs > 0 and self._alt_acqf:
            # for testing / comparison
            return self._alt_acqf(X)

        # batch_size = X.shape[0]
        q = X.shape[-2]
        assert len(self.X_samples) > q, "You should use num_X_samples > q"
        num_dim = X.shape[-1]
        num_obs = len(self.model.train_inputs[0])

        mvn = self.model.posterior(X)
        model_f = self.model.condition_on_observations(X=X, Y=mvn.mean)
        if self._lengthscale_correction:
            model_f.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + max(num_obs, q))) ** (1.0 / num_dim)

        mvn_f = model_f.posterior(self.X_samples, observation_noise=True)
        self.mvn_f = mvn_f

        if self.ttype == "mvar":
            # I-Optimality
            var_f = mvn_f.variance.squeeze()
            m = var_f.mean(dim=-1)
            return -m
        elif self.ttype == "msvar":
            # faster appx. G-Optimality
            var_f = mvn_f.variance.squeeze()
            m = var_f.mean(dim=-1)
            s = var_f.std(dim=-1)
            return -(m + self.beta * s)
        elif self.ttype == "mxi":
            mx = mvn.mean.squeeze(dim=-1).amax(dim=-1)
            std_f = mvn_f.stddev.mean(dim=-1)
            return mx - std_f
        elif self.ttype == "srsg":
            Y_arms = self.get_posterior_samples(mvn).squeeze(-1)
            mx = Y_arms.squeeze(dim=-1).amax(dim=-1).mean(dim=0)
            std_f = mvn_f.stddev.mean(dim=-1)
            return mx - std_f
        elif self.ttype == "maxvar":
            # G-Optimality
            var_f = mvn_f.variance.squeeze()
            return -var_f.max(dim=-1).values
        elif self.ttype == "mcmax":
            Y = self.get_posterior_samples(mvn_f).squeeze(-1)
            return -Y.amax(dim=-1).mean(dim=0)
        elif self.ttype == "varvar":
            var_f = mvn_f.variance.squeeze()
            if self._num_obs == 0:
                return -var_f.max(dim=-1).values
            m = var_f.mean(dim=-1)
            return -m
        elif self.ttype in ["ei", "ei2", "msei"]:
            mu_f = mvn_f.mean.squeeze()
            sd_f = mvn_f.stddev.squeeze()
            if self.ttype in ["ei", "msei"]:
                y_0 = self.Y_max
            elif self.ttype == "ei2":
                y_0 = self.Y_best
            else:
                assert False, self.ttype
            u = _scaled_improvement(mu_f, sd_f, y_0)
            af = sd_f * _ei_helper(u)
            if self.ttype.startswith("ms"):
                return -(af.mean(dim=-1) + af.std(dim=-1))
            else:
                return -af.mean(dim=-1)
            # return -af.max(dim=-1).values
        elif self.ttype in ["ucb", "msucb"]:
            mu_f = mvn_f.mean.squeeze()
            sd_f = mvn_f.stddev.squeeze()
            af = mu_f + self.beta_ucb * sd_f
            if self.ttype == "ucb":
                # return -af.max(dim=-1).values
                return -af.mean(dim=-1)
            elif self.ttype == "msucb":
                return -(af.mean(dim=-1) + af.std(dim=-1))
            else:
                assert False
        elif self.ttype == "sr":
            Y = self.get_posterior_samples(mvn_f).squeeze(-1)
            return -Y.squeeze(-1).max(dim=0).values.max(dim=-1).values
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
