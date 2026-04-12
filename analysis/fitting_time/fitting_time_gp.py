"""GP / SVGP / Vecchia fit helpers for surrogate timing benchmarks."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch import Tensor


def fit_exact_gp(train_x: Tensor, train_y: Tensor, x_test: Tensor) -> tuple[float, Tensor, Tensor]:
    import gpytorch
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    with gpytorch.settings.max_cholesky_size(2000):
        t_0 = time.perf_counter()
        fit_gpytorch_mll(mll)
        elapsed = time.perf_counter() - t_0
    gp.eval()

    x_e = x_test.to(device=train_x.device, dtype=train_x.dtype)
    with torch.no_grad():
        post = gp.posterior(x_e, observation_noise=True)
        y_hat = post.mean
        pred_var = post.variance
    return elapsed, y_hat, pred_var


def _fit_svgp(
    train_x: Tensor,
    train_y: Tensor,
    x_test: Tensor,
    *,
    inducing_points: int | None,
) -> tuple[float, Tensor, Tensor]:
    import gpytorch
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.approximate_gp import SingleTaskVariationalGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from gpytorch.mlls import VariationalELBO

    ty = train_y
    if ty.ndim == 1:
        ty = ty.unsqueeze(-1)

    n = train_x.shape[-2]

    # Standardize inside the model so ``posterior`` untransforms to the original y scale.
    # Passing pre-standardized ``train_y`` without an outcome transform left means in z-space
    # while metrics use raw ``y_test`` (NRMSE / LogLik were wrong).
    if ty.shape[0] <= 1:
        svgp = SingleTaskVariationalGP(train_x, ty, inducing_points=inducing_points)
    else:
        svgp = SingleTaskVariationalGP(
            train_x,
            ty,
            outcome_transform=Standardize(m=ty.shape[-1]),
            inducing_points=inducing_points,
        )
    svgp.to(train_x)
    mll = VariationalELBO(svgp.likelihood, svgp.model, num_data=n)
    with gpytorch.settings.max_cholesky_size(2000):
        t_0 = time.perf_counter()
        fit_gpytorch_mll(
            mll,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"step_limit": 150},
        )
        elapsed = time.perf_counter() - t_0
    svgp.eval()

    x_e = x_test.to(device=train_x.device, dtype=train_x.dtype)
    with torch.no_grad():
        post = svgp.posterior(x_e, observation_noise=True)
        y_hat = post.mean
        pred_var = post.variance
    return elapsed, y_hat, pred_var


def fit_svgp_default(train_x: Tensor, train_y: Tensor, x_test: Tensor) -> tuple[float, Tensor, Tensor]:
    """SVGP with BoTorch defaults (``inducing_points=None`` → ~25% of ``N`` inducing points)."""
    return _fit_svgp(train_x, train_y, x_test, inducing_points=None)


def fit_svgp_linear(train_x: Tensor, train_y: Tensor, x_test: Tensor) -> tuple[float, Tensor, Tensor]:
    """SVGP with ``inducing_points = min(N, 100)`` (integer count, allocator picks locations)."""
    n = int(train_x.shape[-2])
    m = min(n, 100)
    return _fit_svgp(train_x, train_y, x_test, inducing_points=m)


def _vecchia_nan_out(
    x_test: Tensor,
) -> tuple[float, Tensor, Tensor]:
    nan_t = torch.full(
        (x_test.shape[0], 1),
        float("nan"),
        dtype=x_test.dtype,
        device=x_test.device,
    )
    return float("nan"), nan_t, nan_t


def fit_vecchia(train_x: Tensor, train_y: Tensor, x_test: Tensor) -> tuple[float, Tensor, Tensor]:
    """RF Vecchia GP (``pyvecch``): same recipe as :class:`optimizer.vecchia_designer.VecchiaDesigner`.

    Uses float32 on CPU inside ``pyvecch``; returns ``y_hat`` and ``pred_var`` on ``train_x``'s
    device/dtype.     Same recipe as :class:`~optimizer.vecchia_designer.VecchiaDesigner`; attempts ``pyvecch`` on
    all platforms. On macOS we set ``KMP_DUPLICATE_LIB_OK`` before import to reduce duplicate
    OpenMP runtime issues with faiss.

    Set ``YUBO_ALLOW_PYVECCH_ON_DARWIN=0`` (or ``false``) to skip on macOS only and return NaNs
    (e.g. if your environment still crashes on import).
    """
    if sys.platform == "darwin" and os.environ.get("YUBO_ALLOW_PYVECCH_ON_DARWIN", "").lower() in {
        "0",
        "false",
        "no",
    }:
        return _vecchia_nan_out(x_test)

    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        from gpytorch.constraints import Interval
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.means import ZeroMean
        from pyvecch.input_transforms import Identity
        from pyvecch.models import RFVecchia
        from pyvecch.nbrs import ExactOracle
        from pyvecch.prediction import IndependentRF
        from pyvecch.training import fit_model
    except (ImportError, OSError, RuntimeError):
        return _vecchia_nan_out(x_test)

    X = train_x.detach().float().cpu().contiguous()
    Y = train_y.detach().float().cpu().contiguous()
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    n_obs = X.shape[0]
    if n_obs < 2:
        return _vecchia_nan_out(x_test)

    y_mean = Y.mean()
    y_std = Y.std()
    if float(y_std.item()) <= 0.0:
        y_std = torch.tensor(1.0, dtype=Y.dtype, device=Y.device)
    z = ((Y - y_mean) / y_std).squeeze(-1)

    dim = int(X.shape[-1])
    # Same as VecchiaBO ``bo_loop.ipynb``: ``m = int(7.2 * np.log10(n)**2)``; see
    # https://github.com/feji3769/VecchiaBO/blob/master/notebooks/bo_loop.ipynb
    m_nbrs = max(1, int(7.2 * np.log10(max(2, n_obs)) ** 2))
    covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
    mean_module = ZeroMean()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    neighbor_oracle = ExactOracle(X, z, m_nbrs)
    prediction_strategy = IndependentRF()
    input_transform = Identity(d=dim)

    model = RFVecchia(
        covar_module,
        mean_module,
        likelihood,
        neighbor_oracle,
        prediction_strategy,
        input_transform,
    )

    train_batch_size = int(np.minimum(n_obs, 128))
    try:
        t_0 = time.perf_counter()
        fit_model(
            model,
            train_batch_size=train_batch_size,
            n_window=50,
            maxiter=100,
            rel_tol=5e-3,
        )
        elapsed = time.perf_counter() - t_0
        model.update_transform()
        model.eval()
        model.likelihood.eval()
    except (ImportError, OSError, RuntimeError, ValueError, ArithmeticError):
        return _vecchia_nan_out(x_test)

    X_test = x_test.detach().float().cpu().contiguous()
    try:
        with torch.no_grad():
            mvn_dist = model.posterior(X_test)
            mu_z = mvn_dist.mean.squeeze(0)
            var_z = mvn_dist.covariance_matrix.diagonal(dim1=-2, dim2=-1).squeeze(0).clamp_min(1e-30)
        mu_y = mu_z.unsqueeze(-1) * y_std + y_mean
        var_y = var_z.unsqueeze(-1) * (y_std**2)
        y_hat = mu_y.to(device=train_x.device, dtype=train_x.dtype)
        pred_var = var_y.to(device=train_x.device, dtype=train_x.dtype)
    except (RuntimeError, ValueError, ArithmeticError):
        return _vecchia_nan_out(x_test)

    return elapsed, y_hat, pred_var
