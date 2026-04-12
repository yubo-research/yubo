"""Shared ENN regression fit for UHD imputer and seed selector (single implementation)."""

from __future__ import annotations

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def fit_enn_regressor_on_points(
    x_list: list,
    y_list: list,
    *,
    k: int,
) -> tuple[float, float, EpistemicNearestNeighbors, object]:
    """Fit EpistemicNearestNeighbors + hyperparameters on (x, y) observations."""
    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    y_mean = float(y.mean())
    ys = float(y.std())
    y_std = ys if ys > 0 else 1.0
    y_norm = (y - y_mean) / y_std
    model = EpistemicNearestNeighbors(
        x,
        y_norm[:, None],
        None,
        scale_x=False,
        index_driver=ENNIndexDriver.FLAT,
    )
    rng = np.random.default_rng(0)
    params = enn_fit(
        model,
        k=int(k),
        num_fit_candidates=200,
        num_fit_samples=200,
        rng=rng,
    )
    return y_mean, y_std, model, params
