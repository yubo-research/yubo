import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
)
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from botorch.utils import t_batch_mode_transform

# from IPython.core.debugger import set_trace
from scipy.stats import multivariate_normal
from torch import Tensor
from torch.quasirandom import SobolEngine


class AcqIEIG(MCAcquisitionFunction):
    """Integrated Expected Information Gain

    TODO: better to not use weights b/c the sample is from p*(x)
    """

    def __init__(
        self,
        model: Model,
        num_X_samples: int = 256,
        num_px_weights: int = 4096,
        num_px_mc: int = 4096,
        num_mcmc: int = 10,
        p_all_type: str = "all",
        num_fantasies: int = 0,
        num_Y_samples: int = 1024,
        num_noisy_maxes: int = 10,
        q_ts=None,
        no_log=False,
        fantasies_only=True,
        use_des=False,
        no_weights=False,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))
        if num_fantasies > 0:
            self.sampler_fantasies = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        self.num_fantasies = num_fantasies
        self.num_Y_num_f = num_Y_samples * num_fantasies
        self.num_px_mc = num_px_mc
        self.num_px_weights = num_px_weights
        self.c_time = 0.0
        self._no_log = no_log
        self._fantasies_only = fantasies_only
        self._use_des = use_des

        if len(self.model.train_inputs[0]) == 0:
            X_samples = self._sobol_samples(num_X_samples)
        else:
            X_samples = self._sample_X(num_noisy_maxes, num_X_samples, num_mcmc, p_all_type)

        assert len(X_samples) >= num_X_samples, len(X_samples)
        if len(X_samples) != num_X_samples:
            i = np.random.choice(np.arange(len(X_samples)), size=(num_X_samples,), replace=False)
            self.X_samples = X_samples[i]
        else:
            self.X_samples = X_samples

        with torch.no_grad():
            self.p_max = self._calc_p_max(self.model, self.X_samples, num_px=self.num_px_weights)
            self.weights = self.p_max.clone().type(self.X_samples.dtype)

            if no_weights:
                th = 0.1 / len(self.X_samples)
                i = np.where(self.weights.detach().numpy() >= th)[0]
                self.weights = 1.0 + 0.0 * self.weights[i]
                self.X_samples = self.X_samples[i]
                assert len(self.weights) > 0

            self.weights = self.weights / self.weights.sum()

            if q_ts is not None:
                i = np.random.choice(np.arange(len(self.X_samples)), p=self.weights, size=(q_ts,))
                self.X_cand = torch.atleast_2d(X_samples[i])

    def _likelihoods(self, samples):
        xs = []
        for s in samples:
            assert s.x.min() >= 0 and s.x.max() <= 1, s.x
            xs.append(torch.tensor(s.x))
        X = torch.stack(xs)
        if len(X.shape) == 1:
            X = X[:, None]
        else:
            X = torch.atleast_2d(X)

        return 1e-9 + self._calc_p_max(self.model, X, self.num_px_mc)[:, None].mean(axis=1)

    def _cem_sample_X(self, num_X_samples, q_ts):
        # X = self.model.train_inputs[0]
        # num_dim = X.shape[-1]

        x_opt = self._find_max(self._get_noisy_model()).detach().numpy().flatten()
        with torch.no_grad():
            num_samples = 10
            for i_outer in range(1):
                # cem = CEMNIW(mu_0=0.5 * np.ones(shape=(num_dim,)), scale_0=0.03, known_mu=False)
                cem = CEMNIW(mu_0=x_opt, scale_0=0.01, known_mu=False)
                for i_inner in range(100):
                    samples = cem.ask(num_samples)
                    likelihoods = self._likelihoods(samples)
                    if likelihoods is None:
                        print("NOPE:", i_outer, i_inner, cem.estimate_mu_cov())
                        num_samples *= 3
                        break
                    cem.tell(likelihoods, samples)  # , n_keep=num_samples // 3)
                else:
                    break
            else:
                assert False, ("Could not fit p*(x)", num_samples)

            mu_est, cov_est = cem.estimate_mu_cov()
            # assert np.all(mu_est == x_opt), (mu_est, x_opt)

            X = self.model.train_inputs[0]
            if q_ts:
                rv = multivariate_normal(mean=mu_est, cov=cov_est)
                x = rv.rvs(size=(10 * q_ts,))
                x = x[x.min(axis=1) >= 0]
                x = x[x.max(axis=1) <= 1]
                assert len(x) >= q_ts, (mu_est, cov_est)
                x = x[:q_ts, :]
                self.X_cand = torch.tensor(x, dtype=X.dtype)

            qmcn = MultivariateNormalQMCEngine(
                torch.tensor(mu_est),
                torch.tensor(np.diag(cov_est)),
            )
            X_samples = []
            while len(X_samples) < num_X_samples:
                x = qmcn.draw(num_X_samples)
                x = x[x.min(axis=1).values >= 0]
                x = x[x.max(axis=1).values <= 1]
                X_samples.extend(list(x))

            X_samples = torch.stack(X_samples).type(X.dtype)[:num_X_samples].detach()
            self.weights = torch.tensor(rv.pdf(X_samples.numpy()), dtype=X.dtype)

        return X_samples

    def _sample_X(self, num_noisy_maxes, num_X_samples, num_mcmc, p_all_type):
        if num_noisy_maxes == 0:
            models = [self.model]
        else:
            models = [self._get_noisy_model() for _ in range(num_noisy_maxes)]
        x_max = []
        for model in models:
            x = self._find_max(model).detach()
            x_max.append(x)
        x_max = torch.cat(x_max, axis=0)
        no2 = num_X_samples
        n = int(no2 / len(x_max) + 1)
        # X_samples = torch.cat((X_samples, torch.tile(x_max, (n, 1))), axis=0)
        X_samples = torch.tile(x_max, (n, 1))
        assert len(X_samples) >= num_X_samples, (len(X_samples), num_X_samples, no2)

        # burn in
        for _ in range(num_mcmc):
            X_samples = self._mcmc(models, X_samples, eps=0.1, p_all_type=p_all_type)

        # collect paths
        X_all = []
        for _ in range(num_mcmc):
            X_samples = self._mcmc(models, X_samples, eps=0.1, p_all_type=p_all_type)
            X_all.append(X_samples)
        X_all = torch.cat(X_all, axis=0)
        return X_all

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

        x_cand, _ = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return x_cand

    def _sobol_samples(self, num_X_samples):
        X_0 = self.model.train_inputs[0]
        num_dim = X_0.shape[-1]
        sobol_engine = SobolEngine(num_dim, scramble=True)
        return sobol_engine.draw(num_X_samples).to(X_0.device).type(X_0.dtype)

    def _mcmc(self, models, X, eps, p_all_type):
        # TODO: keep the whole path (after warm-up)
        # TODO: NUTS
        with torch.no_grad():
            X_new = X + eps * torch.randn(size=X.shape)
            i = np.where((X_new < 0) | (X_new > 1))[0]
            X_new[i] = torch.rand(size=(len(i), X.shape[-1])).type(X.dtype)
            assert torch.all((X_new >= 0) & (X_new <= 1)), X_new
            X_both = torch.cat((X, X_new), axis=0)
            if p_all_type == "all":
                p_all = 1e-9 + torch.cat([self._calc_p_max(m, X_both, self.num_px_mc)[:, None] for m in models], axis=1).mean(axis=1)
            elif p_all_type == "random":
                p_all = 1e-9 + self._calc_p_max(np.random.choice(models), X_both, self.num_px_mc)[:, None].mean(axis=1)
            elif p_all_type == "self":
                p_all = 1e-9 + self._calc_p_max(self.model, X_both, self.num_px_mc)[:, None].mean(axis=1)
            else:
                assert False, p_all_type
            p = p_all[: len(X)]
            p_new = p_all[len(X) :]

            a = p_new / p
            u = torch.rand(size=(len(X),))
            i = u <= a
            X[i] = X_new[i]
        return X

    def _calc_p_max(self, model, X, num_px):
        posterior = model.posterior(X, posterior_transform=self.posterior_transform, observation_noise=True)
        Y = posterior.sample(torch.Size([num_px])).squeeze(dim=-1)  # num_Y_samples x b x len(X)
        return self._calc_p_max_from_Y(Y)

    def _calc_p_max_from_Y(self, Y):
        is_best = torch.argmax(Y, dim=-1)
        idcs, counts = torch.unique(is_best, return_counts=True)
        p_max = torch.zeros(Y.shape[-1])
        p_max[idcs] = counts / Y.shape[0]
        return p_max

    def _calc_soft_p_max_from_Y(self, Y):
        beta = 20
        sm = torch.exp(beta * Y)
        # Y ~ num_Y_samples x num_fantasies x b x num_X_samples
        if self._fantasies_only:
            norm = sm.sum(dim=-1).sum(dim=0)
            norm = norm.unsqueeze(-1).unsqueeze(0)
        else:
            norm = sm.sum(dim=-1).sum(dim=-1).sum(dim=0)
            norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        sm = sm / norm

        p_max = sm.mean(dim=0)
        return p_max

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self.to(device=X.device)

        if self._no_log:

            def log(x):
                return x

        else:

            def log(x):
                return torch.log(x)

        if self.num_fantasies == 0:
            mvn = self.model.posterior(X, observation_noise=True)
            # Y = self.get_posterior_samples(mvn).squeeze(dim=-1)  # num_Y_samples x b x q

            model_t = self.model.condition_on_observations(X=X, Y=mvn.mean)  # TODO: noise=observation_noise?
            mvn_t = model_t.posterior(self.X_samples, observation_noise=True)
            Y_f = self.get_posterior_samples(mvn_t).squeeze(dim=-1)  # num_Y_samples x b x num_X_samples
            Y_f = Y_f.unsqueeze(1)  # num_Y_samples x 1 x b x num_X_samples
            # H = (self.weights * log(Y.std(dim=0))).sum(dim=-1)
        else:
            model_f = self.model.fantasize(X=X, sampler=self.sampler_fantasies, observation_noise=True)
            mvn_f = model_f.posterior(self.X_samples, observation_noise=True)
            Y_f = self.get_posterior_samples(mvn_f).squeeze(dim=-1)  # num_Y_samples x num_fantasies x b x num_X_samples
            # Y = Y_f.reshape(self.num_Y_num_f, Y_f.shape[-2], Y_f.shape[-1])

        if self._use_des:
            # DES
            p_max = self._calc_soft_p_max_from_Y(Y_f)
            assert p_max.min() >= 0 and p_max.max() <= 1
            # weights = (1 + p_max) / (1 + self.weights)
            weights = p_max / torch.maximum(p_max, self.weights)
            weights = weights / weights.mean()
            H = -(weights * torch.log(p_max)).mean(dim=-1).mean(dim=0)
        else:
            if self._no_log:
                H = (self.weights * Y_f.var(dim=0)).sum(dim=-1).mean(dim=0)
            else:
                # IOPT
                H = (self.weights * log(Y_f.std(dim=0))).sum(dim=-1).mean(dim=0)

        return -H
