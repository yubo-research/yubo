from dataclasses import dataclass

import numpy as np
import torch
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from scipy.stats import multivariate_normal


@dataclass
class Sample:
    p: np.array
    x: np.array


def qmc_normal_sample(mu, cov, num_samples=1):
    qmcn = MultivariateNormalQMCEngine(
        torch.tensor(mu),
        torch.tensor(np.diag(cov)),
    )
    return qmcn.draw(num_samples)


def mk_normal_samples(mu_covs, num_samples, bound, qmc=False):
    samples = []
    while True:
        for mu, cov in mu_covs:
            rv_norm = multivariate_normal(
                mean=mu,
                cov=cov,
            )

            if qmc:
                x = qmc_normal_sample(mu, cov)
            else:
                x = rv_norm.rvs(size=(1,))
            if len(mu) == 1:
                x = np.array([x])
            if bound:
                if x.min() < 0 or x.max() > 1:
                    continue
            samples.append(
                Sample(
                    p=rv_norm.pdf(x),
                    x=x,
                )
            )
            if len(samples) == num_samples:
                return samples
