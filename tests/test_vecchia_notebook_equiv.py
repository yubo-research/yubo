import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.generation import MaxPosteriorSampling
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from pyvecch.input_transforms import Identity
from pyvecch.models import RFVecchia
from pyvecch.nbrs import ExactOracle
from pyvecch.prediction import IndependentRF
from pyvecch.training import fit_model
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(state, model, X, Y, batch_size, n_candidates=None, acqf="ts"):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    assert acqf == "ts"
    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=X.dtype, device=X.device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=X.dtype, device=X.device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=X.device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def test_vecchia_notebook_equivalence_first_5_steps():
    torch.manual_seed(17)
    np.random.seed(17)

    tkwargs = {"dtype": torch.float32, "device": torch.device("cpu")}
    fun = Ackley(dim=20, negate=True).to(**tkwargs)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds

    batch_size = 4
    n_init = 2 * dim

    def eval_objective(x):
        return fun(unnormalize(x, fun.bounds))

    sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    X = sobol.draw(n=n_init).to(**tkwargs)
    Y = torch.tensor([eval_objective(x_) for x_ in X], **tkwargs)

    state = TurboState(dim, batch_size=batch_size)

    training_settings = {"n_window": 50, "maxiter": 100, "rel_tol": 5e-3}

    outputs = []
    for _ in range(5):  # produce lengths 44,48,52,56,60
        n = X.shape[0]
        m = int(7.2 * np.log10(n) ** 2)
        train_batch_size = int(np.minimum(n, 128))

        z = (Y - Y.mean()) / (Y.std() if Y.std() > 0 else 1.0)

        covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
        mean_module = ZeroMean()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

        neighbor_oracle = ExactOracle(X, z, m)
        prediction_stategy = IndependentRF()
        input_transform = Identity(d=dim)
        model = RFVecchia(covar_module, mean_module, likelihood, neighbor_oracle, prediction_stategy, input_transform)

        fit_model(model, train_batch_size=train_batch_size, **training_settings)
        model.update_transform()
        model.eval()
        model.likelihood.eval()

        X_next = generate_batch(
            state=state,
            model=model,
            X=X,
            Y=z,
            batch_size=batch_size,
            n_candidates=min(5000, max(2000, 200 * dim)),
            acqf="ts",
        ).squeeze(0)

        Y_next = torch.tensor([eval_objective(x_) for x_ in X_next], **tkwargs)

        state = update_state(state=state, Y_next=Y_next)

        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)

        outputs.append((len(X), state.best_value, state.length))

    expected = [
        (44, -1.26e01, 8.00e-01),
        (48, -1.20e01, 8.00e-01),
        (52, -1.15e01, 8.00e-01),
        (56, -1.11e01, 8.00e-01),
        (60, -1.04e01, 8.00e-01),
    ]

    for (n, best, tr), (en, ebest, etr) in zip(outputs, expected):
        assert n == en
        # compare with tolerance since floats
        print("COMPARE:", n, en, best, ebest, tr, etr)
        assert abs(best - ebest) < 5e-2
        assert abs(tr - etr) < 1e-6
