"""Notebook-oriented surrogate fit routines: time the fit step, then predict at ``x_test``."""

from __future__ import annotations

import time

import numpy as np
from torch import Tensor

__all__ = [
    "fit_dngo",
    "fit_enn",
    "fit_exact_gp",
    "fit_smac_rf",
    "fit_svgp_default",
    "fit_svgp_linear",
]

# Matches ``0.1 * torch.randn`` observation noise in ``benchmark_synthetic_sine_surrogates``.
_SYNTHETIC_OBS_VAR = 0.1**2


def fit_enn(train_x: np.ndarray, train_y: np.ndarray, x_test: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit
    from enn.enn.enn_params import PosteriorFlags

    train_yvar = np.full_like(train_y, _SYNTHETIC_OBS_VAR)
    n_obs = train_x.shape[0]
    k = min(25, max(1, n_obs - 1))
    enn_model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    t_0 = time.perf_counter()
    enn_params = enn_fit(
        enn_model,
        k=k,
        num_fit_candidates=100,
        num_fit_samples=min(10, n_obs),
        rng=np.random.default_rng(0),
    )
    elapsed = time.perf_counter() - t_0

    x_t = np.asarray(x_test, dtype=np.float64)
    post = enn_model.posterior(x_t, params=enn_params, flags=PosteriorFlags(observation_noise=False))
    y_hat = np.asarray(post.mu, dtype=np.float64).reshape(-1, 1)
    se = np.asarray(post.se, dtype=np.float64).reshape(-1, 1)
    # Match ``observation_noise=False``: predictive variance for scoring is epistemic only.
    pred_var = se**2
    return elapsed, y_hat, pred_var


def fit_smac_rf(train_x: np.ndarray, train_y: np.ndarray, x_test: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from analysis.fitting_time.smac_rf import SMACRFConfig, SMACRFSurrogate

    y1 = train_y.reshape(-1)
    smac_rf = SMACRFSurrogate(
        SMACRFConfig(
            n_trees=10,
            ratio_features=0.5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            seed=0,
        )
    )
    t_0 = time.perf_counter()
    smac_rf.fit(train_x, y1)
    elapsed = time.perf_counter() - t_0

    mean, var = smac_rf.predict(x_test)
    y_hat = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
    pred_var = np.asarray(var, dtype=np.float64).reshape(-1, 1) + _SYNTHETIC_OBS_VAR
    return elapsed, y_hat, pred_var


def fit_dngo(train_x: np.ndarray, train_y: np.ndarray, x_test: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    # "the optimal
    # architecture is a deep and narrow network with 3 hidden
    # layers and approximately 50 hidden units per layer"
    # Scalable Bayesian Optimization Using Deep Neural Networks
    # Snoek et al., ICML 2015
    # https://arxiv.org/pdf/1502.05700
    d_in = int(train_x.shape[1])
    n_obs = int(train_x.shape[0])
    # Paper used ~50 units at low D; for larger D (e.g. notebook ``D=10``) increase width and
    # training so the net is not under-parameterized for sparse high-dimensional data.
    width = max(50, min(384, 50 + 28 * d_in))
    num_epochs = int(min(1200, max(450, 350 + 28 * d_in + n_obs // 5)))
    if d_in >= 8:
        num_epochs = min(1200, num_epochs + 200)
    weight_decay = 1e-3 if d_in <= 5 else 5e-4
    num_middle_layers = 4 if d_in >= 8 else 3

    dngo = DNGOSurrogate(
        DNGOConfig(
            hidden_width=width,
            feature_dim=width,
            num_middle_layers=num_middle_layers,
            num_epochs=num_epochs,
            learning_rate=3e-3,
            weight_decay=weight_decay,
            seed=0,
        )
    )
    t_0 = time.perf_counter()
    dngo.fit(train_x, train_y)
    elapsed = time.perf_counter() - t_0

    mean, var = dngo.predict(x_test)
    y_hat = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
    pred_var = np.asarray(var, dtype=np.float64).reshape(-1, 1)
    return elapsed, y_hat, pred_var


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
        post = gp.posterior(x_e)
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
        post = svgp.posterior(x_e)
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
