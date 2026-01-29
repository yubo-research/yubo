import numpy as np
import scipy.stats as ss
import torch


def boot_means(X, num_boot):
    w = torch.tensor(
        ss.dirichlet(np.ones(len(X))).rvs(num_boot), device=X.device, dtype=X.dtype
    )
    return (w.unsqueeze(-1) * X).sum(axis=1)
