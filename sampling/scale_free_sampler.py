import numpy as np
import torch

from sampling.log_uniform import np_log_uniform


def scale_free_sampler(X_0: torch.Tensor, b_raasp=True):
    num_dim = X_0.shape[-1]

    # RASSSP
    # https://proceedings.mlr.press/v238/rashidi24a/rashidi24a.pdf
    # Cylindrical Thompson Sampling for High-Dimensional Bayesian Optimization
    # B. Rashidi, K. Johnstonbaugh, C. Gao
    # https://proceedings.neurips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf
    # Scalable Global Optimization via Local Bayesian Optimization
    # D. Eriksson, M. Pearce, J. Gardner
    if b_raasp:
        d = np.ceil(np_log_uniform(1, num_dim)).flatten()
        d = int(d[0])
        i = np.arange(num_dim)
        np.random.shuffle(i)
        i = i[:d]
    else:
        i = np.arange(num_dim)
        d = num_dim
    X_t = X_0.clone()
    X_t[0, i] = torch.rand(size=(1, d)).to(X_0)

    # Stagger perturbation
    # https://openreview.net/forum?id=YEuObecAhQ&noteId=YEuObecAhQ
    # Efficient Thompson Sampling for Bayesian Optimization
    # D. Sweet, S. A. Jadhav
    # s = np_log_uniform(1e-6, 1)
    s = np.random.uniform()  # TEST
    return X_0 + s * (X_t - X_0)
