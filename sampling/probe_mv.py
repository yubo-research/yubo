import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_normal_samples

from sampling.bootstrap import boot_means


def proposal_stagger(
    X_0, sigma_min, sigma_max, num_samples, device=None, dtype=torch.double
):
    num_dim = len(X_0)
    l_min_sigma = np.log(sigma_min)
    l_max_sigma = np.log(sigma_max)
    u = l_min_sigma + (l_max_sigma - l_min_sigma) * torch.rand(
        size=torch.Size([num_samples, num_dim]), device=device, dtype=dtype
    )
    sigma = torch.exp(u)

    normal = draw_sobol_normal_samples(
        d=num_dim,
        n=num_samples,
        device=device,
        dtype=dtype,
    )
    X = X_0 + sigma * normal
    # sigma are all equally probable
    pi = torch.exp(-(normal**2 / 2).sum(dim=1))
    pi = pi / pi.sum()
    return pi, X


def proposal_normal(X_0, sigma, num_samples, device=None, dtype=torch.double):
    normal = draw_sobol_normal_samples(
        d=len(sigma),
        n=num_samples,
        device=device,
        dtype=dtype,
    )
    X = X_0 + torch.atleast_2d(sigma) * normal
    pi = torch.exp(-(normal**2).sum(dim=1))
    return pi / pi.sum(), X


class ProbeMV:
    def __init__(self, X_0, sigma_min=None, sigma_max=None):
        # The proposal distribution should have heavier tails than the
        #  target. Go for 100x larger sigma than you think you might find.
        self._X_0 = X_0
        num_dim = len(self._X_0)
        if sigma_min is None:
            self._sigma_min = torch.tensor([1e-6] * num_dim)
        else:
            self._sigma_min = sigma_min
        if sigma_max is None:
            self._sigma_max = torch.tensor([10] * num_dim)
        else:
            self._sigma_max = sigma_max
        self._conv = 1000
        self._mu_std_est = torch.tensor([100] * len(self._X_0))

    def convergence_criterion(self):
        return self._conv

    def sigma_estimate(self):
        return self._mu_std_est

    def ask(self, num_samples, X_and_p_target=None, num_boot=1000):
        if X_and_p_target is not None:
            X, p_target = X_and_p_target
            sigma_min, sigma_max = self._recalculate_sigma_range(X, p_target, num_boot)
        else:
            sigma_min = self._sigma_min
            sigma_max = self._sigma_max
        if True:
            self._pi, X = proposal_stagger(
                X_0=self._X_0,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                num_samples=num_samples,
            )
        else:
            self._pi, X = proposal_normal(
                self._X_0,
                sigma=(sigma_min + sigma_max) / 2,
                num_samples=num_samples,
            )
        assert self._pi.min() > 0, self._pi
        return X

    def _recalculate_sigma_range(self, X, p_target, num_boot):
        p_target = p_target / p_target.sum()
        assert p_target.max() > 0, p_target
        w = p_target / self._pi
        w = w / w.sum()

        dev = X - self._X_0
        # mean should be zero
        # mean_est = (w[:, None] * dev).sum(dim=0)

        wd2 = w[:, None] * dev**2
        std_est = torch.sqrt(len(wd2) * boot_means(wd2, num_boot))
        l_std_est = torch.log(std_est)
        se_l_std_est = l_std_est.std(dim=0)
        mu_l_std_est = l_std_est.mean(dim=0)

        sigma_min = torch.exp(mu_l_std_est - 2 * se_l_std_est)
        sigma_max = torch.exp(mu_l_std_est + 2 * se_l_std_est)

        self._mu_std_est = torch.sqrt((w[:, None] * dev**2).sum(dim=0))
        # self._conv = float(((sigma_max - sigma_min) / mu_std_est).max())
        self._conv = (w**2).mean() / (w.mean() ** 2) - 1

        return sigma_min, sigma_max
