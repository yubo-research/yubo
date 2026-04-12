###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

from copy import deepcopy

import gpytorch
import numpy as np
import torch

from .gp import train_gp
from .turbo_1_core import CandidatesResult, compute_y_cand
from .utils import make_sobol_candidates


def create_candidates(self, X, fX, length, n_training_steps, hypers):
    """Generate candidates assuming X has been scaled to [0,1]^d."""
    assert X.min() >= 0.0 and X.max() <= 1.0

    mu, sigma = np.median(fX), fX.std()
    sigma = 1.0 if sigma < 1e-6 else sigma
    fX = (deepcopy(fX) - mu) / sigma

    if len(X) < self.min_cuda:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = self.device, self.dtype

    with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
        X_torch = torch.tensor(X).to(device=device, dtype=dtype)
        y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
        if self._surrogate_type == "original":
            gp = train_gp(
                train_x=X_torch,
                train_y=y_torch,
                use_ard=self.use_ard,
                num_steps=n_training_steps,
                hypers=hypers,
            )
            hypers = gp.state_dict()

    x_center = X[fX.argmin().item(), :][None, :]
    if self._surrogate_type == "original":
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
    else:
        weights = np.ones(shape=x_center.shape)

    weights = weights / weights.mean()
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

    X_cand = make_sobol_candidates(
        dim=self.dim,
        n_cand=self.n_cand,
        x_center=x_center,
        lb=lb,
        ub=ub,
        device=device,
        dtype=dtype,
    )

    if len(X_cand) < self.min_cuda:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = self.device, self.dtype

    y_cand = compute_y_cand(
        self,
        X=X,
        fX=fX,
        X_cand=X_cand,
        mu=mu,
        sigma=sigma,
        gp=gp if self._surrogate_type == "original" else None,
        device=device,
        dtype=dtype,
    )

    del X_torch, y_torch

    return CandidatesResult(X_cand=X_cand, y_cand=y_cand, hypers=hypers)


def select_candidates(self, X_cand, y_cand):
    if self._surrogate_type.startswith("enn-mu-"):
        assert self.batch_size == 1, self.batch_size
        i = np.argmin(y_cand.mu)
        X_next = X_cand[[i], :]
    elif self._surrogate_type.startswith("enn-se-"):
        assert self.batch_size == 1, self.batch_size
        i = np.argmax(y_cand.se)
        X_next = X_cand[[i], :]
    elif self._surrogate_type.startswith("enn-rand-"):
        y_cand.se = np.random.uniform(size=y_cand.se.shape)
        X_next = arms_from_pareto_fronts(X_cand, y_cand, self.batch_size)
    elif self._surrogate_type.startswith("enn-"):
        X_next = arms_from_pareto_fronts(X_cand, y_cand, self.batch_size)
    elif self._surrogate_type == "none":
        from acq.acq_util import torch_random_choice

        X_next = torch_random_choice(X_cand, self.batch_size, replace=False)
    else:
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
    return X_next


def arms_from_pareto_fronts(x_cand, mvn, num_arms):
    i = np.argsort(mvn.mu, axis=0).flatten()
    x_cand = x_cand[i]
    se = mvn.se[i]

    i_all = list(range(len(se)))
    i_keep = []

    while len(i_keep) < num_arms:
        se_max = -1e99
        i_front = []
        for i in i_all:
            if se[i] >= se_max:
                i_front.append(i)
                se_max = se[i]
        if len(i_keep) + len(i_front) <= num_arms:
            i_keep.extend(i_front)
        else:
            i_keep.extend(np.random.choice(i_front, size=num_arms - len(i_keep), replace=False))
        i_all = sorted(set(i_all) - set(i_front))

    i_keep = np.array(i_keep)
    x_arms = x_cand[i_keep]

    assert len(x_arms) == num_arms, (len(x_arms), num_arms)
    return x_arms
