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

import sys
from copy import deepcopy

import numpy as np
import torch

from .turbo_1_candidates import create_candidates, select_candidates
from .turbo_1_core import init_counters_and_tr, init_hypers, validate_init_args
from .utils import from_unit_cube, latin_hypercube, to_unit_cube, turbo_adjust_length


class Turbo1Standard:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        *,
        surrogate_type="original",
    ):
        validate_init_args(
            lb,
            ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            dtype=dtype,
        )

        self._surrogate_type = "original" if surrogate_type == "gp" else surrogate_type

        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        init_hypers(self)
        init_counters_and_tr(self, batch_size=batch_size)

        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device(device)
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        turbo_adjust_length(self, fX_next)

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        return create_candidates(self, X, fX, length, n_training_steps, hypers)

    def _select_candidates(self, X_cand, y_cand):
        return select_candidates(self, X_cand, y_cand)

    def _evaluate(self, X):
        assert len(X) % self.batch_size == 0, (len(X), self.batch_size)
        Y = []
        while len(Y) < len(X):
            x = X[len(Y) : len(Y) + self.batch_size]
            assert len(x) == self.batch_size, (len(x), self.batch_size, len(X), len(Y))
            Y.extend(self.f(x))
        assert len(Y) == len(X), (len(Y), len(X))
        return np.array(Y)[:, None]

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            self._restart()

            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = self._evaluate(X_init)

            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            while self.n_evals < self.max_evals and self.length >= self.length_min:
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)
                fX = deepcopy(self._fX).ravel()

                X_cand, y_cand, _ = self._create_candidates(
                    X,
                    fX,
                    length=self.length,
                    n_training_steps=self.n_training_steps,
                    hypers={},
                )
                X_next = self._select_candidates(X_cand, y_cand)
                del y_cand

                X_next = from_unit_cube(X_next, self.lb, self.ub)

                fX_next = self._evaluate(X_next)

                self._adjust_length(fX_next)

                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    n_evals, fbest = self.n_evals, fX_next.min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()

                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))


Turbo1 = Turbo1Standard
