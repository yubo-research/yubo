import numpy as np
import torch

# from botorch.utils.sampling import draw_sobol_normal_samples
from sampling.bootstrap import boot_means
from third_party.torch_truncnorm.TruncatedNormal import TruncatedNormal

_ygtiw = "You got the indexing wrong"


def _proposal_stagger(X_0, sigma_min, sigma_max, num_samples_per_dimension):
    device = X_0.device
    dtype = X_0.dtype

    X_0 = X_0.flatten()

    num_dim = len(X_0)
    l_min_sigma = torch.log(sigma_min)
    l_max_sigma = torch.log(sigma_max)

    # rng = torch.Generator(device=device).manual_seed(seed)

    num_samples = num_samples_per_dimension * num_dim
    u = l_min_sigma + (l_max_sigma - l_min_sigma) * torch.rand(
        size=torch.Size([num_samples_per_dimension, num_dim]),
        device=device,
        dtype=dtype,
        # generator=rng,
    )
    sigma = torch.exp(u)

    i = torch.arange(num_samples)
    # all 0, then all 1, then ...
    j = torch.repeat_interleave(torch.arange(num_dim), repeats=(num_samples_per_dimension))
    assert torch.all(j[:num_samples_per_dimension] == 0), _ygtiw

    loc = torch.zeros(size=(num_samples_per_dimension * num_dim,), dtype=dtype, device=device)
    loc[i] = X_0[j]

    scale = torch.zeros(size=(num_samples_per_dimension * num_dim,), dtype=dtype, device=device)
    scale[i] = sigma.T.flatten()
    assert scale[1] == sigma[1, 0], _ygtiw
    assert num_dim == 1 or scale[num_samples_per_dimension] == sigma[0, 1], _ygtiw

    tn = TruncatedNormal(
        loc=loc,
        scale=scale,
        a=torch.zeros_like(loc),
        b=torch.ones_like(loc),
        # TODO: seed
    )

    pi_and_x = tn.p_and_sample((1,))
    pi = pi_and_x.pi.flatten()
    X_perturbed_dimension = pi_and_x.X.flatten()
    assert X_perturbed_dimension.dtype == dtype, (X_perturbed_dimension.dtype, dtype)
    assert pi.dtype == dtype, (pi.dtype, dtype)

    X_perturbed_dimension = X_perturbed_dimension.squeeze()
    pi = pi.squeeze()
    assert X_perturbed_dimension.shape == (num_samples,), X_perturbed_dimension.shape
    assert pi.shape == (num_samples,)

    X = torch.tile(X_0, dims=(num_samples, 1))
    assert X.shape == (num_samples, num_dim), _ygtiw

    X[i, j] = X_perturbed_dimension

    assert not torch.any((X.flatten() < 0) | (X.flatten() > 1)), (X.min(), X.max())

    pi = _norm_by_dim(pi, num_dim, num_samples_per_dimension)
    # pi = torch.maximum(torch.tensor(1e-5 / len(pi)), pi)

    assert torch.all(torch.isfinite(pi)), (sigma_min, sigma_max, pi)
    assert torch.all(pi > 0), (sigma_min, sigma_max, pi)
    return pi, X


def _norm_by_dim(p, num_dim, num_samples_per_dimension):
    # Treat each dimension as a separate problem.
    # We're only solving them in parallel b/c we
    #  think we can be more efficient with
    #  parallel sampling(ex., w/C++, threads, GPU).

    p_r = p.reshape(num_dim, num_samples_per_dimension)
    assert num_dim == 1 or p_r[1, 0] == p[num_samples_per_dimension]
    norm = p_r.sum(axis=1, keepdim=True)
    p_r = p_r / norm
    i = torch.where(norm == 0)[0]
    p_r[i, :] = 1 / p_r.shape[1]
    p = p_r.reshape(num_dim * num_samples_per_dimension)
    assert num_dim == 1 or p_r[1, 0] == p[num_samples_per_dimension]
    return p


class _StaggerISSampler:
    def __init__(self, X, p_tn):
        assert len(p_tn) == len(X), len(p_tn) == len(X)
        self._X = X
        self._p_tn = p_tn

    def ask(self):
        return self._X

    def tell(self, p_target, w_max=10):
        assert len(p_target) == len(self._X), len(p_target) == len(self._X)
        p_target = p_target / p_target.sum()
        w = p_target / self._p_tn
        w = torch.min(torch.tensor(w_max), w)
        w = w / w.sum()

    def sample(self, num_samples):
        i = np.arange(len(self._X))
        i = np.random.choice(i, size=(num_samples,), p=self._w)
        return self._X[i]


class StaggerIS:
    def __init__(self, X_0, sigma_min=None, sigma_max=None):
        # The proposal distribution should have heavier tails than the
        #  target. Go for 100x larger sigma than you think you might find.
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
        self._conv = 1000
        self._mu_std_est = torch.tensor([self._sigma_max.max()] * len(self._X_0))

    def convergence_criterion(self):
        return self._conv

    def sigma_estimate(self):
        return self._mu_std_est

    def importance_weights(self, X, p_target):
        p_target = p_target / p_target.sum()
        sigma = self.sigma_estimate()
        p_normal = torch.exp(-((X - self._X_0) ** 2) / (2 * sigma**2)).flatten()
        p_normal = p_normal / p_normal.sum()
        return p_target / p_normal

    def _mk_1d(self, x, num_base_samples):
        x = torch.tile(torch.atleast_2d(x.flatten()).T, dims=(num_base_samples,)).flatten()
        assert x[0] == x[1]
        if self._num_dim > 1:
            assert x[num_base_samples] == x[num_base_samples + 1]
        return x

    def _unmk_1d(self, x, num_base_samples):
        return torch.reshape(x, (self._num_dim, num_base_samples)).T

    def sampler(self, num_base_samples):
        sigma = self._mk_1d(self.sigma_estimate(), num_base_samples)
        X_0 = self._mk_1d(self._X_0, num_base_samples)

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
        self._pi, X = _proposal_stagger(
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

        # Use bootstrap to sample a bunch (num_boot) of std_ests
        #  so that we can find sigma_min and sigma_max.
        std_est = torch.sqrt(wd2.shape[0] * boot_means(wd2, num_boot))
        l_std_est = torch.log(std_est)
        se_l_std_est = l_std_est.std(dim=0)
        mu_l_std_est = l_std_est.mean(dim=0)

        sigma_min = torch.exp(mu_l_std_est - 2 * se_l_std_est)
        sigma_max = torch.exp(mu_l_std_est + 2 * se_l_std_est)

        self._mu_std_est = torch.sqrt(wd2.sum(dim=0))
        w = w.reshape(self._num_dim, num_samples_per_dimension).T
        # .max() is difficult when num_dim is high b/c p{max is good enough} ~ exp(-num_dim)
        if True:
            c = sigma_max / sigma_min - 1
            if self._num_dim > 3:
                self._conv = min(c.max(), float(c.mean() + 2 * c.std()))
            else:
                self._conv = c.max()
        else:
            c = (w**2).mean(dim=0) / (w.mean(dim=0) ** 2) - 1
            if self._num_dim > 3:
                self._conv = float(c.mean() + 2 * c.std())
            else:
                self._conv = float(c.max())

        assert torch.all(torch.isfinite(sigma_min)), sigma_min
        assert torch.all(torch.isfinite(sigma_max)), sigma_max
        sigma_max = torch.minimum(torch.tensor(10), sigma_max)
        # assert torch.all(sigma_max < 1e4), (sigma_max, l_std_est, wd2)

        return sigma_min, sigma_max
