import numpy as np
import torch

from sampling.bootstrap import boot_means
from sampling.sampling_util import var_of_var
from third_party.torch_truncnorm.TruncatedNormal import TruncatedNormal

from .stagger_distribution import StaggerDistribution


class _StaggerISSampler:
    def __init__(self, X, p_tn):
        assert len(p_tn) == len(X), len(p_tn) == len(X)
        self._X = X
        self._p_tn = p_tn

    def ask(self):
        return self._X

    def tell(self, p_target, w_max=100):
        assert len(p_target) == len(self._X), len(p_target) == len(self._X)
        p_target = p_target / p_target.sum()
        w = p_target / self._p_tn
        w = torch.min(torch.tensor(w_max), w)
        self._w = w / w.sum()

    def sample(self, num_samples):
        i = np.arange(len(self._X))
        i = np.random.choice(i, size=(num_samples,), p=self._w)
        return self._X[i]


class StaggerIS:
    def __init__(self, X_0, sigma_min=None, sigma_max=None, seed=None):
        # The proposal distribution should have heavier tails than the
        #  target.
        self._X_0 = X_0.flatten()
        self.dtype = self._X_0.dtype
        self.device = self._X_0.device
        self._num_dim = len(self._X_0)

        if sigma_min is None:
            self._sigma_min = torch.tensor(
                [1e-6] * self._num_dim,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._sigma_min = sigma_min
        if sigma_max is None:
            self._sigma_max = torch.tensor(
                [10] * self._num_dim,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._sigma_max = sigma_max

        self._s_min_now = sigma_min
        self._s_max_now = sigma_max
        self._conv_d_sigma = 1000
        self._conv_R = 1
        self._mu_std_est = torch.tensor([self._sigma_max.max()] * len(self._X_0))
        self._seed = seed

    def convergence_criterion(self):
        return self._conv_R

    def convergence_criterion_d_sigma(self):
        return self._conv_d_sigma

    def sigma_estimate(self):
        return self._mu_std_est

    def importance_weights(self, X, p_target):
        p_target = p_target / p_target.sum()
        sigma = self.sigma_estimate()
        p_normal = torch.exp(-((X - self._X_0) ** 2) / (2 * sigma**2)).flatten()
        if p_normal.sum() > 0:
            p_normal = p_normal / p_normal.sum()
        else:
            p_normal = 0 * p_normal
        pi = p_target / p_normal
        pi[p_normal < 1e-6 * p_normal.mean()] = 0
        pi = torch.min(torch.tensor(100), pi)
        # if pi.sum() > 0:
        #     pi = pi / pi.sum()
        # else:
        #     pi = 1 + 0 * pi
        # try:
        #     assert not torch.any(torch.isnan(pi))
        # except:
        #     breakpoint()
        return pi

    def _reshape_1d(self, x, num_base_samples):
        x = torch.tile(torch.atleast_2d(x.flatten()).T, dims=(num_base_samples,)).flatten()
        assert x[0] == x[1]
        if self._num_dim > 1:
            assert x[num_base_samples] == x[num_base_samples + 1]
        return x

    def _unmk_1d(self, x, num_base_samples):
        return torch.reshape(x, (self._num_dim, num_base_samples)).T

    def sampler(self, num_base_samples):
        sigma = self._reshape_1d(self.sigma_estimate(), num_base_samples)
        X_0 = self._reshape_1d(self._X_0, num_base_samples)

        tn = TruncatedNormal(
            loc=X_0,
            scale=sigma,
            a=torch.zeros_like(X_0),
            b=torch.ones_like(X_0),
        )
        pi_and_X = tn.p_and_sample((1,))
        p_tn = pi_and_X.pi.flatten()
        X = pi_and_X.X.flatten()
        X = self._unmk_1d(X, num_base_samples)
        p_tn = self._unmk_1d(p_tn, num_base_samples)
        p_tn = torch.cumprod(p_tn, dim=1)[:, -1]
        if p_tn.sum() == 0:
            p_tn[:] = 1 / len(p_tn)
        else:
            p_tn = p_tn / p_tn.sum()
        p_tn = p_tn / p_tn.sum()
        assert X.shape == (num_base_samples, self._num_dim)
        assert p_tn.shape == (num_base_samples,)
        return _StaggerISSampler(X, p_tn)

    def ask(self, num_samples_per_dimension, X_and_p_target=None, num_boot=1000):
        if X_and_p_target is not None:
            X, p_target = X_and_p_target
            sigma_min, sigma_max = self._recalculate_sigma_range(X, p_target, num_boot)
        else:
            sigma_min = self._sigma_min
            sigma_max = self._sigma_max

        # print("S:", sigma_min, sigma_max)
        self._pi, X = StaggerDistribution(
            X_0=self._X_0,
            num_samples_per_dimension=num_samples_per_dimension,
            seed=self._seed,
        ).propose(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        self._s_min_now = sigma_min
        self._s_max_now = sigma_max
        return X

    def _recalculate_sigma_range(self, X, p_target, num_boot):
        num_samples_per_dimension = len(X) / self._num_dim
        assert num_samples_per_dimension == int(num_samples_per_dimension)
        num_samples_per_dimension = int(num_samples_per_dimension)

        p_target = StaggerDistribution.norm_by_dim(p_target, self._num_dim, num_samples_per_dimension)
        assert torch.all(torch.isfinite(p_target)), p_target
        assert torch.all(self._pi > 0), self._pi
        w = p_target / self._pi
        assert w.sum() > 0, w
        w = StaggerDistribution.norm_by_dim(w, self._num_dim, num_samples_per_dimension)
        assert torch.all(torch.isfinite(w)), w

        i = torch.arange(self._num_dim * num_samples_per_dimension)
        j = torch.repeat_interleave(torch.arange(self._num_dim), repeats=(num_samples_per_dimension))
        dev = (X - self._X_0)[i, j]

        # mean should be zero
        # mean_est = (w[:, None] * dev).sum(dim=0)

        wd2 = w * dev**2
        wd2 = wd2.reshape(self._num_dim, num_samples_per_dimension).T

        self._mu_std_est = torch.sqrt(wd2.sum(dim=0))
        # Use bootstrap to sample a bunch (num_boot) of std_ests
        #  so that we can find sigma_min and sigma_max.
        std_est = torch.sqrt(wd2.shape[0] * boot_means(wd2, num_boot))

        # k = 2
        # se = std_est.std(dim=0)
        # sigma_min = self._mu_std_est - k * se
        # sigma_max = self._mu_std_est + k * se

        # Log-sigma is more symmetric than sigma
        #  and may be negative (sigma may not).
        l_std_est = torch.log(std_est)
        mu_l_std_est = l_std_est.mean(dim=0)
        se_l_std_est = l_std_est.std(dim=0)

        k = 2
        sigma_min = torch.exp(mu_l_std_est - k * se_l_std_est)
        sigma_max = torch.exp(mu_l_std_est + k * se_l_std_est)

        w = w.reshape(self._num_dim, num_samples_per_dimension).T
        # .max() is difficult when num_dim is high b/c p{max is good enough} ~ exp(-num_dim)
        c = sigma_max / sigma_min - 1
        if self._num_dim > 3:
            self._conv_d_sigma = min(c.max(), float(c.mean() + 2 * c.std()))
        else:
            self._conv_d_sigma = c.max()

        c = (w**2).mean(dim=0) / (w.mean(dim=0) ** 2) / num_samples_per_dimension - 1
        if self._num_dim > 3:
            self._conv_R = float(c.mean() + 2 * c.std())
        else:
            self._conv_R = float(c.max())

        assert torch.all(torch.isfinite(sigma_min)), sigma_min
        assert torch.all(torch.isfinite(sigma_max)), sigma_max
        sigma_min = torch.maximum(torch.as_tensor(self._sigma_min), sigma_min)
        sigma_max = torch.minimum(torch.as_tensor(self._sigma_max), sigma_max)
        # assert torch.all(sigma_max < 1e4), (sigma_max, l_std_est, wd2)

        return sigma_min, sigma_max
