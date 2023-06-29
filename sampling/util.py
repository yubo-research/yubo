from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal


@dataclass
class Sample:
    p: np.array
    x: np.array


def mk_normal_samples(mu_covs, num_samples):
    samples = []
    while len(samples) < num_samples:
        for mu, cov in mu_covs:
            rv_norm = multivariate_normal(
                mean=mu,
                cov=cov,
            )

            x = rv_norm.rvs(size=(1,))
            if len(mu) == 1:
                x = np.array([x])
            if x.min() < 0 or x.max() > 1:
                continue
            samples.append(
                Sample(
                    p=rv_norm.pdf(x),
                    x=x,
                )
            )

    return samples[:num_samples]
