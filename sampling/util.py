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

    return [Sample(x=xx, p=pp) for xx, pp in zip(x, p)]
