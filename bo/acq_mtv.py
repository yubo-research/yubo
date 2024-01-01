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
from torch.quasirandom import SobolEngine


class AcqMTV(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        num_X_samples,
        ttype,
        beta=0,
        num_mcmc=5,
        num_Y_samples=1,
        sample_type="mh",
        x_max_type="find_max",
        alt_acqf=None,
        lengthscale_correction=None,
        eps_0=0.1,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.num_mcmc = num_mcmc
        self.num_X_samples = num_X_samples
        self.ttype = ttype
        self.beta = beta
        self._alt_acqf = alt_acqf
        self._lengthscale_correction = lengthscale_correction
        self._eps_0 = eps_0
        self.eps_interior = torch.tensor(1e-6)
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
            if x_max_type == "find_max":
                self.X_max = self._find_max()
            elif x_max_type == "best_obs":
                Y = self.model.posterior(self.model.train_inputs[0]).mean.squeeze(dim=-1)
                i = np.argmax(Y.detach())
                self.X_max = self.model.train_inputs[0][i, :][:, None].T
            else:
                assert False, ("Unknown x_max_type", x_max_type)
            self.Y_max = self.model.posterior(self.X_max).mean
            if len(self.model.train_targets) > 0:
                i = torch.argmax(self.model.train_targets)
                self.Y_best = self.model.posterior(self.model.train_inputs[0][i][:, None].T).mean
            else:
                self.Y_best = self.Y_max

            if sample_type == "mh":
                with torch.inference_mode():
                    self.X_samples = self._sample_maxes_mh(sobol_engine, num_X_samples, prop_type="met")
            elif sample_type == "hnr":
                with torch.inference_mode():
                    self.X_samples = self._sample_maxes_mh(sobol_engine, num_X_samples, prop_type="hnr")
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

        Y_cand = self.model.posterior(x_cand).mean
        if len(self.model.train_targets) > 0:
            i = torch.argmax(self.model.train_targets)
            Y_tgt = self.model.posterior(self.model.train_inputs[0][i][:, None].T).mean
            if Y_tgt > Y_cand:
                x_cand = self.model.train_inputs[0][i, :][:, None].T

        return x_cand

    def _sample_maxes_mh(self, sobol_engine, num_X_samples, prop_type):
        X_max = torch.maximum(self.eps_interior, torch.minimum(1 - self.eps_interior, self.X_max))
        X = torch.tile(X_max, (num_X_samples, 1))

        eps = 1
        eps_good = False
        num_changed = 0
        max_iterations = 10 * self.num_mcmc
        frac_changed = None
        for _ in range(max_iterations):
            if prop_type == "met":
                i, X_1 = self._met_propose(X, eps)
            elif prop_type == "hnr":
                i, X_1 = self._hnr_propose(X, eps)
            X[i] = X_1[i]
            frac_changed = (1.0 * i).mean().item()
            # print("FC:", eps, eps_good, frac_changed)
            if frac_changed > 0.50 - 1e-5:
                eps_good = True
            elif frac_changed < 0.40:
                eps_good = False

            if not eps_good:
                eps = eps / np.sqrt(10.0)
            else:
                num_changed += 1
                if num_changed == self.num_mcmc:
                    break
        else:
            print(
                f"WARNING: Could not determine eps in {max_iterations} iterations. Was at eps = {eps} and num_changed = {num_changed}. Last frac_changed = {frac_changed}"
            )

        return X

    def _find_bounds(self, X, u, eps_bound):
        X = X.detach().numpy()
        u = u.detach().numpy()
        num_chains = X.shape[0]
        l_low = np.zeros(shape=(num_chains, 1))
        l_high = np.ones(shape=(num_chains, 1))

        def _accept(X):
            return (X.min(axis=1) >= 0) & (X.max(axis=1) <= 1)

        while (l_high - l_low).max() > eps_bound:
            l_mid = (l_low + l_high) / 2
            X_mid = X + l_mid * u
            a = _accept(X_mid)
            l_low[a] = l_mid[a]
            l_high[~a] = l_mid[~a]
            # print ("B:", (l_high - l_low).max(), l_low.mean())

        return l_low.flatten()

    def _hnr_propose(self, X, eps):
        from scipy.stats import truncnorm

        num_chains = X.shape[0]
        num_dim = X.shape[1]

        for _ in range(5):
            # random direction, u
            u = torch.randn(size=(num_chains, num_dim))
            u = u / torch.sqrt((u**2).sum(axis=1, keepdims=True))

            # Find bounds along u
            eps_bound = min(eps, float(self.eps_interior)) / 100
            llambda_plus = self._find_bounds(X, u, eps_bound)
            llambda_minus = self._find_bounds(X, -u, eps_bound)
            min_length = (llambda_plus - -(llambda_minus)).min()
            # print ("BOUNDS:", (llambda_plus - -(llambda_minus)))
            if min_length > 0:
                break
        else:
            assert False, "Could not find a perturbation direction"

        # 1D perturbation
        rv = truncnorm(-llambda_minus / eps, llambda_plus / eps, scale=eps)
        X_1 = X + torch.tensor(rv.rvs(num_chains))[:, None] * u
        assert torch.all((X_1.min(dim=1).values >= 0) & (X_1.max(dim=1).values <= 1)), "Perturbation failed"

        # Metropolis update
        X_both = torch.cat((X, X_1), dim=0)
        mvn = self.model.posterior(X_both, observation_noise=True)
        Y_both = mvn.sample(torch.Size([1])).squeeze(-1).squeeze(0)
        Y = Y_both[: len(X)]
        Y_1 = Y_both[len(X) :]
        b_met_accept = (Y_1 > Y).flatten()

        return b_met_accept, X_1

    def _met_propose(self, X, eps):
        X_1 = X + eps * torch.randn(size=X.shape)
        # Joint sample for all X -- initial and proposed
        X_both = torch.cat((X, X_1), dim=0)
        mvn = self.model.posterior(X_both, observation_noise=True)
        Y_both = mvn.sample(torch.Size([1])).squeeze(-1).squeeze(0)
        Y = Y_both[: len(X)]
        Y_1 = Y_both[len(X) :]

        # P{maximizer | out-out-bounds} == 0
        # return (X_1.min(dim=1).values >= 0) & (X_1.max(dim=1).values <= 1) & (Y_1 > Y).flatten(), X_1
        i_good = (X_1.min(dim=1).values >= -1e-6) & (X_1.max(dim=1).values <= 1 + 1e-6) & (Y_1 > Y).flatten()
        X_1[i_good] = torch.maximum(torch.tensor(0.0), torch.minimum(torch.tensor(1.0), X_1[i_good]))
        return i_good, X_1

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

        mvn_a = self.model.posterior(X)
        model_f = self.model.condition_on_observations(X=X, Y=mvn_a.mean)
        if self._lengthscale_correction is not None:
            # num_obs = 0 ==> 1 "bin", size of whole box
            # num_obs = 1 ==> 2 "bins", each size of half the box
            # etc
            if self._lengthscale_correction == "type_0":
                model_f.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + max(num_obs, q))) ** (1.0 / num_dim)
            elif self._lengthscale_correction == "type_1":
                model_f.covar_module.base_kernel.lengthscale *= ((1 + num_obs) / (1 + num_obs + q)) ** (1.0 / num_dim)
            else:
                assert False, self._lengthscale_correction

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
        elif self.ttype == "maxvar":
            # G-Optimality
            var_f = mvn_f.variance.squeeze()
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
