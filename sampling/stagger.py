import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_normal_samples

from sampling.bootstrap import boot_means

# TODO: Leave this in torch and try on GPU.

# TODO: Try just using sigma*random_sign.
# TODO: Compare to choosing proposal from a normal distribution.


def proposal_stagger(X_0, sigma_min, sigma_max, num_samples_per_dimension, device=None, dtype=torch.double):
    num_dim = len(X_0)
    l_min_sigma = np.log(sigma_min)
    l_max_sigma = np.log(sigma_max)
    u = l_min_sigma + (l_max_sigma - l_min_sigma) * torch.rand(size=torch.Size([num_samples_per_dimension, num_dim]), device=device, dtype=dtype)
    sigma = torch.exp(u)

    normal_1d = draw_sobol_normal_samples(
        d=1,
        n=num_samples_per_dimension * num_dim,
        device=device,
        dtype=dtype,
    ).squeeze(-1)

    stagger = torch.zeros(size=(num_samples_per_dimension * num_dim, num_dim), dtype=dtype, device=device)
    j = torch.repeat_interleave(torch.arange(num_dim), repeats=(num_samples_per_dimension))
    i = torch.arange(stagger.shape[0])
    stagger[i, j] = normal_1d
    stagger[i, j] *= sigma.T.reshape(len(normal_1d))
    X = X_0 + stagger

    # sigma are all equally probable
    pi = torch.exp(-(normal_1d**2 / 2))
    pi = _norm_by_dim(pi, num_dim, num_samples_per_dimension)
    # pi = torch.maximum(torch.tensor(0.00001 / len(pi)), pi)

    assert torch.all(torch.isfinite(pi)), (sigma_min, sigma_max, pi)
    assert torch.all(pi > 0), (sigma_min, sigma_max, pi)
    return pi, X


def _norm_by_dim(p, num_dim, num_samples_per_dimension):
    # Treat each dimension as a separate problem.
    # We're only solving them in parallalel b/c we
    #  think we can be more efficient with
    #  parallel sampling(ex., w/ threads or a GPU).
    p = p.reshape(num_dim, num_samples_per_dimension)
    p = p / p.sum(axis=1, keepdim=True)
    return p.reshape(num_dim * num_samples_per_dimension)


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
        # self._eps_sigma = torch.tensor(1e-9)

    def convergence_criterion(self):
        return self._conv

    def sigma_estimate(self):
        return self._mu_std_est

    def ask(self, num_samples_per_dimension, X_and_p_target=None, num_boot=1000):
        if X_and_p_target is not None:
            X, p_target = X_and_p_target
            sigma_min, sigma_max = self._recalculate_sigma_range(X, p_target, num_boot)
        else:
            sigma_min = self._sigma_min
            sigma_max = self._sigma_max

        # print("S:", sigma_min, sigma_max)
        self._pi, X = proposal_stagger(
            X_0=self._X_0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_samples_per_dimension=num_samples_per_dimension,
        )

        return X

    def _recalculate_sigma_range(self, X, p_target, num_boot):
        num_samples_per_dimension = len(X) / self._num_dim
        assert num_samples_per_dimension == int(num_samples_per_dimension)
        num_samples_per_dimension = int(num_samples_per_dimension)

        p_target = _norm_by_dim(p_target, self._num_dim, num_samples_per_dimension)
        assert torch.all(torch.isfinite(p_target)), p_target
        assert torch.all(self._pi > 0), self._pi
        w = p_target / self._pi
        assert w.sum() > 0, w
        w = _norm_by_dim(w, self._num_dim, num_samples_per_dimension)
        assert torch.all(torch.isfinite(w)), w

        i = torch.arange(self._num_dim * num_samples_per_dimension)
        j = torch.repeat_interleave(torch.arange(self._num_dim), repeats=(num_samples_per_dimension))
        dev = (X - self._X_0)[i, j]

        # mean should be zero
        # mean_est = (w[:, None] * dev).sum(dim=0)

        wd2 = w * dev**2
        wd2 = wd2.reshape(self._num_dim, num_samples_per_dimension).T
        std_est = torch.sqrt(wd2.shape[0] * boot_means(wd2, num_boot))
        l_std_est = torch.log(std_est)
        se_l_std_est = l_std_est.std(dim=0)
        mu_l_std_est = l_std_est.mean(dim=0)

        sigma_min = torch.exp(mu_l_std_est - 2 * se_l_std_est)
        sigma_max = torch.exp(mu_l_std_est + 2 * se_l_std_est)

        self._mu_std_est = torch.sqrt(wd2.sum(dim=0))
        w = w.reshape(self._num_dim, num_samples_per_dimension).T
        self._conv = ((w**2).mean(dim=0) / (w.mean(dim=0) ** 2) - 1).max()

        assert torch.all(torch.isfinite(sigma_min)), sigma_min
        assert torch.all(torch.isfinite(sigma_max)), sigma_max
        assert torch.all(sigma_max < 1e4), (sigma_max, l_std_est, wd2)

        return sigma_min, sigma_max
