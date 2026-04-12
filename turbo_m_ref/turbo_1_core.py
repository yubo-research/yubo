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

from typing import NamedTuple

import gpytorch
import numpy as np
import torch

from .turbo_1_ask_tell_core import _make_candidates


class CandidatesResult(NamedTuple):
    X_cand: np.ndarray
    y_cand: object
    hypers: dict


def validate_init_args(
    lb,
    ub,
    *,
    n_init,
    max_evals,
    batch_size,
    verbose,
    use_ard,
    max_cholesky_size,
    n_training_steps,
    dtype,
):
    assert lb.ndim == 1 and ub.ndim == 1
    assert len(lb) == len(ub)
    assert np.all(ub > lb)
    assert max_evals > 0 and isinstance(max_evals, int)
    assert n_init > 0 and isinstance(n_init, int), n_init
    assert batch_size > 0 and isinstance(batch_size, int)
    assert isinstance(verbose, bool) and isinstance(use_ard, bool)
    assert max_cholesky_size >= 0 and isinstance(max_cholesky_size, (int, np.integer))
    assert n_training_steps >= 30 and isinstance(n_training_steps, int)
    assert max_evals > n_init and max_evals > batch_size
    assert dtype == "float32" or dtype == "float64"


def init_hypers(self):
    self.mean = np.zeros((0, 1))
    self.signal_var = np.zeros((0, 1))
    self.noise_var = np.zeros((0, 1))
    self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))


def init_counters_and_tr(self, *, batch_size):
    self.n_cand = min(100 * self.dim, 5000)
    if self._surrogate_type.startswith("enn-"):
        self.n_cand = max(self.n_cand, 10 * self.batch_size)
    self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
    self.succtol = 3
    self.n_evals = 0
    self.length_min = 0.5**7
    self.length_max = 1.6
    self.length_init = 0.8


def make_X_cand(self, *, x_center, lb, ub, device, dtype):
    return _make_candidates(self, x_center=x_center, lb=lb, ub=ub, device=device, dtype=dtype)


def compute_y_cand(self, *, X, fX, X_cand, mu, sigma, gp, device, dtype):
    if self._surrogate_type == "original":
        gp = gp.to(dtype=dtype, device=device)
        with (
            torch.no_grad(),
            gpytorch.settings.max_cholesky_size(self.max_cholesky_size),
        ):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
        y_cand = mu + sigma * y_cand
        del X_cand_torch, gp
        return y_cand
    if self._surrogate_type == "none":
        return None
    if self._surrogate_type.startswith("enn-"):
        from model.enn import EpistemicNearestNeighbors

        k = int(self._surrogate_type.split("-")[-1])
        enn = EpistemicNearestNeighbors(k=k)
        enn.add(X, fX[:, None])
        return enn.posterior(X_cand)
    raise ValueError(f"Unknown surrogate_type: {self._surrogate_type}")
