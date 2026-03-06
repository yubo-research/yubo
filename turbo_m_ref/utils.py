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

import numpy as np
from torch.quasirandom import SobolEngine


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def make_sobol_candidates(*, dim, n_cand, x_center, lb, ub, device, dtype):
    """Sobol candidate generation with perturbation mask (shared by TuRBO variants)."""
    seed = np.random.randint(int(1e6))
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
    pert = lb + (ub - lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = np.random.rand(n_cand, dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    mask[ind, np.random.randint(0, dim, size=len(ind))] = 1

    X_cand = x_center.copy() * np.ones((n_cand, dim))
    X_cand[mask] = pert[mask]
    return X_cand


def turbo_adjust_length(opt, fX_next):
    if getattr(opt, "length_fixed", False):
        return
    if np.min(fX_next) < np.min(opt._fX) - 1e-3 * abs(float(np.min(opt._fX))):
        opt.succcount += 1
        opt.failcount = 0
    else:
        opt.succcount = 0
        opt.failcount += 1

    if opt.succcount == opt.succtol:
        opt.length = min([2.0 * opt.length, opt.length_max])
        opt.succcount = 0
    elif opt.failcount == opt.failtol:
        opt.length /= 2.0
        opt.failcount = 0
