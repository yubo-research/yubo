"""Shared ENN regression fit for UHD imputer and seed selector (single implementation)."""

from __future__ import annotations

from inspect import signature

import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.turbo.config.enn_index_driver import ENNIndexDriver

_ENN_FIT_PARAMS = set(signature(enn_fit).parameters)


def fit_enn_params(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    num_fit_candidates: int = 200,
    num_fit_samples: int = 200,
    rng: np.random.Generator | None = None,
    yvar: np.ndarray | None = None,
    infer_aleatoric_variance_scale: bool = True,
    params_warm_start=None,
):
    _assert_model_matches_data(model, x, y, yvar)
    fit_rng = rng if rng is not None else np.random.default_rng(0)
    kwargs = {
        "k": int(k),
        "num_fit_candidates": num_fit_candidates,
        "num_fit_samples": num_fit_samples,
        "rng": fit_rng,
        "params_warm_start": params_warm_start,
    }
    if "infer_aleatoric_variance_scale" in _ENN_FIT_PARAMS:
        kwargs["infer_aleatoric_variance_scale"] = infer_aleatoric_variance_scale
    return enn_fit(model, **kwargs)


def _assert_model_matches_data(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    yvar: np.ndarray | None,
) -> None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    model_x, model_y, model_yvar = enn_train_rows(model)
    assert model_x.shape == x_arr.shape
    assert model_y.shape == y_arr.shape
    if yvar is not None:
        assert model_yvar is not None
        assert model_yvar.shape == np.asarray(yvar).shape


def enn_train_rows(
    model: EpistemicNearestNeighbors,
    indices: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ENN training rows across old and current ennbo APIs."""
    if indices is None:
        indices = list(range(len(model)))
    if hasattr(model, "train_rows_at"):
        x, y, yvar = model.train_rows_at(indices)
    else:
        train_yvar = getattr(model, "train_yvar", None)
        x = np.asarray(model.train_x)[indices]
        y = np.asarray(model.train_y)[indices]
        yvar = None if train_yvar is None else np.asarray(train_yvar)[indices]
    yvar_arr = None if yvar is None else np.asarray(yvar, dtype=np.float64)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), yvar_arr


def fit_enn_regressor_on_points(
    x_list: list,
    y_list: list,
    *,
    k: int,
) -> tuple[float, float, EpistemicNearestNeighbors, object]:
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
    params = fit_enn_params(
        model,
        x,
        y_norm,
        k=int(k),
        num_fit_candidates=200,
        num_fit_samples=200,
        rng=np.random.default_rng(0),
    )
    return y_mean, y_std, model, params


def enn_mixin_maybe_fit_inplace(obj) -> None:
    cfg = obj._cfg
    if len(obj._x) < max(2, int(cfg.warmup_real_obs)):
        return
    if obj._enn_params is not None and obj._num_new_since_fit < int(cfg.fit_interval):
        return
    y_mean, y_std, model, params = fit_enn_regressor_on_points(obj._x, obj._y, k=int(cfg.k))
    obj._y_mean = y_mean
    obj._y_std = y_std
    obj._enn_model = model
    obj._enn_params = params
    obj._num_new_since_fit = 0
