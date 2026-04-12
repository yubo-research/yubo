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

from .turbo_1_ask_tell_core import (
    create_candidates,
    init_counters_and_tr,
    init_hypers,
    select_candidates,
    tell_impl,
    validate_init_args,
)
from .utils import from_unit_cube, to_unit_cube, turbo_adjust_length


class Turbo1AskTell:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    x_bounds:
    n_init : Number of initial points (2*dim is recommended), int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(x_bounds=x_bounds)
        turbo1.maximize(max_evals=max_evals)  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        x_bounds,
        length_fixed=None,
        n_init=None,
        batch_size=1,
        verbose=False,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):
        x_bounds = np.asarray(x_bounds)
        lb = x_bounds[:, 0]
        ub = x_bounds[:, 1]
        if n_init is None:
            n_init = 2 * len(lb)

        validate_init_args(
            lb,
            ub,
            n_init=n_init,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            device=device,
            dtype=dtype,
        )

        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        self.n_init = n_init
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        init_hypers(self)

        init_counters_and_tr(self, batch_size=batch_size, length_fixed=length_fixed)

        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        self._X_batch_todo = []
        self._X_batch_done = []
        self._fX_batch_done = []
        self._X_best_so_far = None
        self._X_last = None
        self._y_max = -np.inf

        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()
        self.length = 0
        self._se = None

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        if self.length_fixed:
            self.length = self.length_fixed
        else:
            self.length = self.length_init

    def _adjust_length(self, fX_next):
        turbo_adjust_length(self, fX_next)

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        return create_candidates(self, X, fX, length, n_training_steps, hypers)

    def _select_candidates(self, X_cand, y_cand):
        return select_candidates(self.batch_size, self.dim, X_cand, y_cand)

    def _init(self):
        if len(self._fX) > 0 and self.verbose:
            n_evals, fbest = self.n_evals, self._fX.min()
            print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
            sys.stdout.flush()

        self._restart()

        X_init = np.random.uniform(size=(self.n_init, self.dim))
        X_init = from_unit_cube(X_init, self.lb, self.ub)
        return X_init

    def _next_batch(self):
        if self.length < self.length_min:
            return self._init()

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

        return from_unit_cube(X_next, self.lb, self.ub)

    def _ask(self):
        if len(self._X_batch_todo) == 0:
            self._X_batch_todo = self._next_batch()
            self._X_batch_done = []
            self._fX_batch_done = []

        X_next = self._X_batch_todo[0]
        self._X_batch_todo = self._X_batch_todo[1:]
        return X_next

    def predict(self, predict_best=True):
        if predict_best:
            return self._X_best_so_far
        else:
            return self._X_last

    def ask(self):
        return self._ask()

    def tell(self, y, x, se=None):
        return self._tell(y, x.flatten(), se)

    def _tell(self, y, x, se=None):
        tell_impl(self, y, x, se)

    def maximize(self, f, max_evals):
        assert max_evals > 0 and isinstance(max_evals, int)
        assert max_evals > self.n_init and max_evals > self.batch_size

        while self.n_evals < max_evals:
            X_next = self._ask()
            fX_next = f(X_next)
            self._tell(fX_next, X_next)


Turbo1 = Turbo1AskTell
