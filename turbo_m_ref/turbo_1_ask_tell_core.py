###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch

from .gp import train_gp
from .turbo_types import CandidatesResult as _CandidatesResult
from .turbo_types import StandardizedFX as _StandardizedFX
from .turbo_types import TrustRegion as _TrustRegion
from .utils import make_sobol_candidates, turbo_adjust_length


def validate_init_args(
    lb,
    ub,
    *,
    n_init,
    batch_size,
    verbose,
    use_ard,
    max_cholesky_size,
    n_training_steps,
    device,
    dtype,
):
    assert lb.ndim == 1 and ub.ndim == 1
    assert len(lb) == len(ub)
    assert np.all(ub > lb)
    assert n_init > 0 and isinstance(n_init, int)
    assert batch_size > 0 and isinstance(batch_size, int)
    assert isinstance(verbose, bool) and isinstance(use_ard, bool)
    assert max_cholesky_size >= 0 and isinstance(max_cholesky_size, (int, np.integer))
    assert n_training_steps >= 30 and isinstance(n_training_steps, int)
    assert device == "cpu" or device == "cuda"
    assert dtype == "float32" or dtype == "float64"
    if device == "cuda":
        assert torch.cuda.is_available(), "can't use cuda if it's not available"


def init_hypers(self):
    self.mean = np.zeros((0, 1))
    self.signal_var = np.zeros((0, 1))
    self.noise_var = np.zeros((0, 1))


def init_counters_and_tr(self, *, batch_size, length_fixed):
    self.n_cand = min(100 * self.dim, 5000)
    self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
    self.succtol = 3
    self.n_evals = 0
    self.length_min = 0.5**7
    self.length_max = 1.6
    self.length_init = 0.8
    self.length_fixed = length_fixed


def device_dtype_for(self, n_points: int):
    if n_points < self.min_cuda:
        return torch.device("cpu"), torch.float64
    return self.device, self.dtype


def standardize_fX(fX) -> _StandardizedFX:
    mu, sigma = np.median(fX), fX.std()
    sigma = 1.0 if sigma < 1e-6 else sigma
    return _StandardizedFX(fX=(deepcopy(fX) - mu) / sigma, mu=mu, sigma=sigma)


def train_gp_model(self, X, fX, n_training_steps, hypers, device, dtype):
    with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
        X_torch = torch.tensor(X).to(device=device, dtype=dtype)
        y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
        gp = train_gp(
            train_x=X_torch,
            train_y=y_torch,
            use_ard=self.use_ard,
            num_steps=n_training_steps,
            hypers=hypers,
        )
    del X_torch, y_torch
    return gp


def trust_region_bounds(self, X, fX, gp, length) -> _TrustRegion:
    x_center = X[fX.argmin().item(), :][None, :]
    weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
    weights = weights / weights.mean()
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)
    return _TrustRegion(x_center=x_center, lb=lb, ub=ub)


def sample_candidates(gp, X_cand, *, device, dtype, batch_size, max_cholesky_size):
    gp = gp.to(dtype=dtype, device=device)
    with (
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(max_cholesky_size),
    ):
        X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
        y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([batch_size])).t().cpu().detach().numpy()
    del X_cand_torch, gp
    return y_cand


def _make_candidates(self, *, x_center, lb, ub, device, dtype):
    return make_sobol_candidates(
        dim=self.dim,
        n_cand=self.n_cand,
        x_center=x_center,
        lb=lb,
        ub=ub,
        device=device,
        dtype=dtype,
    )


def create_candidates(self, X, fX, length, n_training_steps, hypers):
    assert X.min() >= 0.0 and X.max() <= 1.0

    z = standardize_fX(fX)
    device, dtype = device_dtype_for(self, len(X))
    gp = train_gp_model(self, X, z.fX, n_training_steps, hypers, device, dtype)

    tr = trust_region_bounds(self, X, z.fX, gp, length)
    X_cand = _make_candidates(
        self,
        x_center=tr.x_center,
        lb=tr.lb,
        ub=tr.ub,
        device=device,
        dtype=dtype,
    )

    device2, dtype2 = device_dtype_for(self, len(X_cand))
    y_cand = sample_candidates(
        gp,
        X_cand,
        device=device2,
        dtype=dtype2,
        batch_size=self.batch_size,
        max_cholesky_size=self.max_cholesky_size,
    )
    y_cand = z.mu + z.sigma * y_cand

    return _CandidatesResult(X_cand=X_cand, y_cand=y_cand, hypers=hypers)


def select_candidates(batch_size, dim, X_cand, y_cand):
    X_next = np.ones((batch_size, dim))
    for i in range(batch_size):
        indbest = np.argmin(y_cand[:, i])
        X_next[i, :] = deepcopy(X_cand[indbest, :])
        y_cand[indbest, :] = np.inf
    return X_next


def tell_impl(self, y, x, se):
    if self._se is None:
        self._se = se
    else:
        assert self._se == se, "Only constant-error modeling supported"

    if y > self._y_max:
        self._y_max = y
        self._X_best_so_far = x
    self._X_last = x

    fX = np.array([-float(y)])
    X = deepcopy(x)

    self.fX = np.concatenate((self.fX, fX[:, None]), axis=0)
    self.X = np.concatenate((self.X, X[:, None].T), axis=0)
    self.n_evals += 1

    self._fX_batch_done.append(fX)
    self._X_batch_done.append(X)

    if len(self._X_batch_todo) == 0:
        self._X_batch_todo = []

        fX_next = np.array(self._fX_batch_done)
        X_next = np.array(self._X_batch_done)

        if len(self._X) == 0:
            self._X = deepcopy(X_next)
            self._fX = deepcopy(fX_next)

        turbo_adjust_length(self, fX_next)

        self._X = np.vstack((self._X, X_next))
        self._fX = np.vstack((self._fX, fX_next))

        if self.verbose and fX_next.min() < self.fX.min():
            n_evals, fbest = self.n_evals, fX_next.min()
            print(f"{n_evals}) New best: {fbest:.4}")
            sys.stdout.flush()
