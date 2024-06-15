from dataclasses import dataclass

import numpy as np
import torch
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from scipy.stats import multivariate_normal
from torch.quasirandom import SobolEngine


@dataclass
class Sample:
    x: np.array
    p: np.array


def var_of_var(w: torch.Tensor, X: torch.Tensor):
    assert torch.abs(w.sum() - 1) < 1e-6, w.sum()
    mu = (w * X**2).sum(dim=0)
    dev = X - mu
    return var_of_var_dev(w, dev)


def var_of_var_dev(w: torch.Tensor, dev: torch.Tensor):
    assert torch.abs(w.sum() - 1) < 1e-6, w.sum()
    mu_2 = (w * dev**2).sum(dim=0)
    mu_4 = (w * dev**4).sum(dim=0)
    n = len(dev)
    return (mu_4 - mu_2**2) / n


def qmc_normal_sample(mu, cov, num_samples=1):
    qmcn = MultivariateNormalQMCEngine(
        torch.tensor(mu),
        torch.tensor(np.diag(cov)),
    )
    return qmcn.draw(num_samples)


def draw_bounded_normal_samples(mu, cov, num_samples, qmc=False):
    rv_norm = multivariate_normal(
        mean=mu,
        cov=cov,
    )

    num_dim = len(mu)

    if qmc:
        x_n = qmc_normal_sample(mu, cov, num_samples).numpy()
        sobol_engine = SobolEngine(num_dim, scramble=True)
        x_u = sobol_engine.draw(num_samples).numpy()
    else:
        x_n = rv_norm.rvs(size=(2 * num_samples,))
        if num_dim == 1:
            x_n = x_n[:, None]
        x_u = np.random.uniform(size=(2 * num_samples, len(mu)))

    p_n = rv_norm.pdf(x_n)
    p_n = p_n / p_n.sum() / 2
    p_u = np.ones(shape=(len(x_u),))
    p_u = p_u / p_u.sum() / 2

    i = np.where((x_n.min(axis=1) >= 0) & (x_n.max(axis=1) <= 1))[0]
    x_n = x_n[i, :]
    p_n = p_n[i]

    x = np.concatenate((x_n, x_u), axis=0)
    p = np.concatenate((p_n, p_u), axis=0)
    p = rv_norm.pdf(x) / p
    p = p / p.sum()

    i = np.random.choice(np.arange(len(x)), p=p, size=(num_samples,))
    x = x[i, :]
    p = p[i]
    p = p / p.sum()

    return x, p


def _xxx_draw_varied_bounded_normal_samples(mus_covs):
    samples = []
    num_dim = len(mus_covs[0][0])
    for mu, cov in mus_covs:
        assert len(mu) == num_dim, (len(mu), num_dim)

        rv_norm = multivariate_normal(
            mean=mu,
            cov=cov,
        )

        x = rv_norm.rvs()
        if num_dim == 1:
            x = np.array([x])
        x = x[:, None]
        if x.min() < 0 or x.max() > 1:
            x = np.random.uniform(size=(1, len(mu)))
        p = rv_norm.pdf(x.flatten())
        samples.append((x, p))

    samples = np.array(samples)
    x = np.stack(samples[:, 0]).squeeze(-1)
    p = samples[:, 1]

    return x, p
