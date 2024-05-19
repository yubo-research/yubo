import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_normal_samples

# TODO: Leave this in torch and try on GPU.

# TODO: Try just using sigma*random_sign.
# TODO: Compare to choosing proposal from a normal distribution.


def proposal_stagger(X_0, sigma_min, sigma_max, num_samples_per_dimension, device=None, dtype=torch.double):
    num_dim = len(X_0)
    l_min_sigma = np.log(sigma_min)
    l_max_sigma = np.log(sigma_max)
    u = l_min_sigma + (l_max_sigma - l_min_sigma) * torch.rand(size=torch.Size([num_samples_per_dimension, num_dim]), device=device, dtype=dtype)
    sigma = torch.exp(u)

    normal_1 = draw_sobol_normal_samples(
        d=1,
        n=num_samples_per_dimension * num_dim,
        device=device,
        dtype=dtype,
    ).squeeze(-1)

    normal = torch.zeros(size=(num_samples_per_dimension * num_dim, num_dim), dtype=dtype, device=device)
    j = torch.repeat_interleave(torch.arange(num_dim), repeats=(num_samples_per_dimension))
    i = torch.arange(normal.shape[0])
    normal[i, j] = sigma.T.reshape(len(normal_1)) * normal_1

    X = X_0 + normal

    # sigma are all equally probable
    pi = torch.exp(-(normal**2 / 2).sum(dim=1))
    return pi / pi.sum(), X


def boot(X):
    n = X.shape[0]
    i = np.random.randint(n, size=(n,))
    return X[i]


class StaggerIS:
    def __init__(self, X_0, sigma_min=None, sigma_max=None):
        # The proposal distribution should have heavier tails than the
        #  target. Go for 100x larger sigma than you think you might find.
        self._X_0 = X_0
        self._num_dim = len(self._X_0)
        if sigma_min is None:
            self._sigma_min = torch.tensor([1e-6] * self._num_dim)
        else:
            self._sigma_min = sigma_min
        if sigma_max is None:
            self._sigma_max = torch.tensor([10] * self._num_dim)
        else:
            self._sigma_max = sigma_max
        self._conv = 1000
        self._mu_std_est = torch.tensor([100] * len(self._X_0))
        self._eps_sigma = torch.tensor(1e-9)

    def convergence_criterion(self):
        return self._conv

    def sigma_estimate(self):
        return self._mu_std_est

    def ask(self, num_samples_per_dimension, X_and_p_target=None, num_boot=30):
        if X_and_p_target is not None:
            X, p_target = X_and_p_target
            sigma_min, sigma_max = self._recalculate_sigma_range(X, p_target, num_boot)
        else:
            sigma_min = self._sigma_min
            sigma_max = self._sigma_max

        print("S:", sigma_min, sigma_max)
        self._pi, X = proposal_stagger(
            X_0=self._X_0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_samples_per_dimension=num_samples_per_dimension,
        )

        # TODO: assert self._pi.min() > 0, self._pi
        # self._pi = torch.maximum(torch.tensor(1e-9), self._pi)

        return X

    def _recalculate_sigma_range(self, X, p_target, num_boot):
        # TODO assert p_target.max() > 0, p_target
        p_target = p_target / p_target.sum()
        w = p_target / self._pi
        w = w / w.sum()

        num_samples = len(X) / self._num_dim
        assert num_samples == int(num_samples)
        num_samples = int(num_samples)

        i = torch.arange(self._num_dim * num_samples)
        j = torch.repeat_interleave(torch.arange(self._num_dim), repeats=(num_samples))
        dev = (X - self._X_0)[i, j]

        # mean should be zero
        # mean_est = (w[:, None] * dev).sum(dim=0)

        wd2 = w * dev**2
        wd2 = wd2.reshape(self._num_dim, num_samples).T
        l_std_est = []
        for _ in range(num_boot):
            s = torch.maximum(self._eps_sigma, torch.sqrt(boot(wd2).sum(dim=0)))
            l_std_est.append(torch.log(s))
        l_std_est = torch.stack(l_std_est)
        se_l_std_est = l_std_est.std(dim=0)
        mu_l_std_est = l_std_est.mean(dim=0)

        sigma_min = torch.exp(mu_l_std_est - 2 * se_l_std_est)
        sigma_max = torch.exp(mu_l_std_est + 2 * se_l_std_est)

        self._mu_std_est = torch.sqrt((wd2).sum(dim=0))
        self._conv = (w**2).mean() / (w.mean() ** 2) - 1

        return sigma_min, sigma_max
