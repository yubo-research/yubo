"""Notebook-oriented surrogate fit routines: time the fit step, then predict at ``x_test``."""

from __future__ import annotations

import time

import numpy as np

from .fitting_time_gp import (
    fit_exact_gp,
    fit_svgp_default,
    fit_svgp_linear,
    fit_vecchia,
)

__all__ = [
    "fit_dngo",
    "fit_enn",
    "fit_enn_hnsw",
    "fit_exact_gp",
    "fit_smac_rf",
    "fit_svgp_default",
    "fit_svgp_linear",
    "fit_vecchia",
]

# Matches ``0.1 * torch.randn`` observation noise in ``benchmark_synthetic_sine_surrogates``.
_SYNTHETIC_OBS_VAR = 0.1**2

# Synthetic benchmarks use a fixed small test set; callers may still pass huge ``x_test`` here.
# Chunk Rust ``posterior`` to cap peak memory for those callers.
_ENN_POSTERIOR_CHUNK = 65_536


def enn_fit_k_and_num_fit_samples(n_obs: int) -> tuple[int, int]:
    n = int(n_obs)
    k_cap = max(1, n - 1)
    k_eff = min(25, k_cap)
    nfs = int(min(max(1, min(10, n)), n))
    return k_eff, nfs


def fit_enn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_test: np.ndarray,
    *,
    k: int | None = None,
    num_fit_samples: int | None = None,
    num_fit_candidates: int = 100,
    rng: np.random.Generator | None = None,
    index_driver=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_params import PosteriorFlags
    from enn.turbo.config.enn_index_driver import ENNIndexDriver

    from optimizer.uhd_enn_fit_helpers import fit_enn_params

    train_yvar = np.full_like(train_y, _SYNTHETIC_OBS_VAR)
    driver = ENNIndexDriver.FLAT if index_driver is None else index_driver
    n_obs = train_x.shape[0]
    k_default, nfs_default = enn_fit_k_and_num_fit_samples(n_obs)
    if k is None:
        k_eff = k_default
    else:
        k_cap = max(1, n_obs - 1)
        k_eff = min(max(1, int(k)), k_cap)
    nfs = num_fit_samples if num_fit_samples is not None else nfs_default
    nfs = int(min(max(1, nfs), n_obs))
    gen = rng if rng is not None else np.random.default_rng(0)

    t_0 = time.perf_counter()
    enn_model = EpistemicNearestNeighbors(
        train_x,
        train_y,
        train_yvar,
        index_driver=driver,
    )
    enn_params = fit_enn_params(
        enn_model,
        train_x,
        train_y,
        k=k_eff,
        num_fit_candidates=int(num_fit_candidates),
        num_fit_samples=nfs,
        rng=gen,
        yvar=train_yvar,
    )
    elapsed = time.perf_counter() - t_0

    x_t = np.asarray(x_test, dtype=np.float64)
    n_test = int(x_t.shape[0])
    chunk = max(1, int(_ENN_POSTERIOR_CHUNK))
    if n_test <= chunk:
        post = enn_model.posterior(x_t, params=enn_params, flags=PosteriorFlags(observation_noise=False))
        y_hat = np.asarray(post.mu, dtype=np.float64).reshape(-1, 1)
        se = np.asarray(post.se, dtype=np.float64).reshape(-1, 1)
    else:
        mu_parts: list[np.ndarray] = []
        se_parts: list[np.ndarray] = []
        for start in range(0, n_test, chunk):
            sl = slice(start, min(start + chunk, n_test))
            post_b = enn_model.posterior(
                x_t[sl],
                params=enn_params,
                flags=PosteriorFlags(observation_noise=False),
            )
            mu_parts.append(np.asarray(post_b.mu, dtype=np.float64).reshape(-1, 1))
            se_parts.append(np.asarray(post_b.se, dtype=np.float64).reshape(-1, 1))
        y_hat = np.concatenate(mu_parts, axis=0)
        se = np.concatenate(se_parts, axis=0)
    # Epistemic variance plus synthetic observation variance so LogLik matches noisy ``y_test``
    # (same ``0.1^2`` as :func:`draw_benchmark_synthetic_xy` and SMAC RF scoring).
    pred_var = se**2 + _SYNTHETIC_OBS_VAR
    return elapsed, y_hat, pred_var


def fit_enn_hnsw(
    train_x: np.ndarray,
    train_y: np.ndarray,
    x_test: np.ndarray,
    **kwargs,
) -> tuple[float, np.ndarray, np.ndarray]:
    from enn.turbo.config.enn_index_driver import ENNIndexDriver

    return fit_enn(
        train_x,
        train_y,
        x_test,
        index_driver=ENNIndexDriver.HNSW,
        **kwargs,
    )


def fit_smac_rf(train_x: np.ndarray, train_y: np.ndarray, x_test: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from analysis.fitting_time.smac_rf import SMACRFConfig, SMACRFSurrogate

    y1 = train_y.reshape(-1)
    t_0 = time.perf_counter()
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

    t_0 = time.perf_counter()
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
    dngo.fit(train_x, train_y)
    elapsed = time.perf_counter() - t_0

    mean, var = dngo.predict(x_test)
    y_hat = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
    pred_var = np.asarray(var, dtype=np.float64).reshape(-1, 1)
    return elapsed, y_hat, pred_var
