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
from typing import NamedTuple

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, to_unit_cube, turbo_adjust_length  # latin_hypercube,


class _CandidatesResult(NamedTuple):
    X_cand: np.ndarray
    y_cand: object
    hypers: dict


class _StandardizedFX(NamedTuple):
    fX: np.ndarray
    mu: float
    sigma: float


class _TrustRegion(NamedTuple):
    x_center: np.ndarray
    lb: np.ndarray
    ub: np.ndarray


def _validate_init_args(
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
    assert max_cholesky_size >= 0 and isinstance(batch_size, int)
    assert n_training_steps >= 30 and isinstance(n_training_steps, int)
    assert device == "cpu" or device == "cuda"
    assert dtype == "float32" or dtype == "float64"
    if device == "cuda":
        assert torch.cuda.is_available(), "can't use cuda if it's not available"


def _init_hypers(self):
    self.mean = np.zeros((0, 1))
    self.signal_var = np.zeros((0, 1))
    self.noise_var = np.zeros((0, 1))


def _init_counters_and_tr(self, *, batch_size, length_fixed):
    self.n_cand = min(100 * self.dim, 5000)
    self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
    self.succtol = 3
    self.n_evals = 0
    self.length_min = 0.5**7
    self.length_max = 1.6
    self.length_init = 0.8
    self.length_fixed = length_fixed


def _device_dtype_for(self, n_points: int):
    if n_points < self.min_cuda:
        return torch.device("cpu"), torch.float64
    return self.device, self.dtype


def _standardize_fX(fX) -> _StandardizedFX:
    mu, sigma = np.median(fX), fX.std()
    sigma = 1.0 if sigma < 1e-6 else sigma
    return _StandardizedFX(fX=(deepcopy(fX) - mu) / sigma, mu=mu, sigma=sigma)


def _train_gp_model(self, X, fX, n_training_steps, hypers, device, dtype):
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


def _trust_region_bounds(self, X, fX, gp, length) -> _TrustRegion:
    x_center = X[fX.argmin().item(), :][None, :]
    weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
    weights = weights / weights.mean()
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)
    return _TrustRegion(x_center=x_center, lb=lb, ub=ub)


def _make_candidates(self, *, x_center, lb, ub, device, dtype):
    seed = np.random.randint(int(1e6))
    sobol = SobolEngine(self.dim, scramble=True, seed=seed)
    pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
    pert = lb + (ub - lb) * pert

    prob_perturb = min(20.0 / self.dim, 1.0)
    mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    mask[ind, np.random.randint(0, self.dim, size=len(ind))] = 1

    X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
    X_cand[mask] = pert[mask]
    return X_cand


def _sample_candidates(gp, X_cand, *, device, dtype, batch_size, max_cholesky_size):
    gp = gp.to(dtype=dtype, device=device)
    with (
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(max_cholesky_size),
    ):
        X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
        y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([batch_size])).t().cpu().detach().numpy()
    del X_cand_torch, gp
    return y_cand


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

        _validate_init_args(
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

        # Save function information
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        _init_hypers(self)
        # self.lengthscales = (
        #    np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))
        # )

        _init_counters_and_tr(self, batch_size=batch_size, length_fixed=length_fixed)

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        self._X_batch_todo = []
        self._X_batch_done = []
        self._fX_batch_done = []
        self._X_best_so_far = None
        self._X_last = None
        self._y_max = -np.inf

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()
        self.length = 0  # trigger first _init() call
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
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        z = _standardize_fX(fX)
        device, dtype = _device_dtype_for(self, len(X))
        gp = _train_gp_model(self, X, z.fX, n_training_steps, hypers, device, dtype)

        tr = _trust_region_bounds(self, X, z.fX, gp, length)
        X_cand = _make_candidates(self, x_center=tr.x_center, lb=tr.lb, ub=tr.ub, device=device, dtype=dtype)

        device2, dtype2 = _device_dtype_for(self, len(X_cand))
        y_cand = _sample_candidates(
            gp,
            X_cand,
            device=device2,
            dtype=dtype2,
            batch_size=self.batch_size,
            max_cholesky_size=self.max_cholesky_size,
        )
        y_cand = z.mu + z.sigma * y_cand

        return _CandidatesResult(X_cand=X_cand, y_cand=y_cand, hypers=hypers)

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next

    def _init(self):
        if len(self._fX) > 0 and self.verbose:
            n_evals, fbest = self.n_evals, self._fX.min()
            print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

        # Generate and evalute initial design points
        # TODO: study this X_init = latin_hypercube(self.n_init, self.dim)
        X_init = np.random.uniform(size=(self.n_init, self.dim))
        X_init = from_unit_cube(X_init, self.lb, self.ub)
        return X_init

    def _next_batch(self):
        if self.length < self.length_min:
            return self._init()

        # Warp inputs
        X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

        # Standardize values
        fX = deepcopy(self._fX).ravel()

        # Create the next batch
        X_cand, y_cand, _ = self._create_candidates(
            X,
            fX,
            length=self.length,
            n_training_steps=self.n_training_steps,
            hypers={},
        )
        X_next = self._select_candidates(X_cand, y_cand)

        # Undo the warping
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
        # This algo restarts, so X_last could
        #  be very bad.
        if predict_best:
            return self._X_best_so_far
        else:
            return self._X_last

    def ask(self):
        return self._ask()

    def tell(self, y, x, se=None):
        return self._tell(y, x.flatten(), se)

    def _tell(self, y, x, se=None):
        if self._se is None:
            self._se = se
        else:
            assert self._se == se, "Only constant-error modeling supported"

        if y > self._y_max:
            self._y_max = y
            self._X_best_so_far = x
        self._X_last = x

        # TuRBO minimizes, but I want to maximize.
        fX = np.array([-float(y)])
        X = deepcopy(x)

        # Append data to the global history
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

            # Update trust region
            self._adjust_length(fX_next)

            # Update budget and append data
            self._X = np.vstack((self._X, X_next))
            self._fX = np.vstack((self._fX, fX_next))

            if self.verbose and fX_next.min() < self.fX.min():
                n_evals, fbest = self.n_evals, fX_next.min()
                print(f"{n_evals}) New best: {fbest:.4}")
                sys.stdout.flush()

    def maximize(self, f, max_evals):
        assert max_evals > 0 and isinstance(max_evals, int)
        assert max_evals > self.n_init and max_evals > self.batch_size

        """Run the full optimization process."""
        # always ends after a complete batch
        while self.n_evals < max_evals:
            X_next = self._ask()

            # Evaluate batch
            fX_next = f(X_next)

            self._tell(fX_next, X_next)


Turbo1 = Turbo1AskTell
