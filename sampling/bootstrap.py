import scipy.stats as ss
import torch


def boot_means(x, num_boot):
    w = torch.tensor(ss.dirichlet(torch.ones(len(x))).rvs(num_boot))
    return (w.unsqueeze(-1) * x).sum(axis=1)
