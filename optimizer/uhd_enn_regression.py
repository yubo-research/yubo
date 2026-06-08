from __future__ import annotations

from typing import Any

import numpy as np


def fit_enn(zs: list[np.ndarray], ys: list[float], enn_k: int):
    from optimizer.uhd_simple_be import _fit_enn

    return _fit_enn(zs, ys, int(enn_k))


def predict_enn(model, params, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from optimizer.uhd_simple_be import _predict_enn

    return _predict_enn(model, params, x)


def fit_if_due(state: dict, cfg, *, require_pair: bool = False) -> None:
    if len(state["zs"]) < int(cfg.be.warmup):
        return
    if require_pair and state.get("phase_since_fit", 0) <= 0:
        return
    if state["params"] is not None and state["new_since_fit"] < int(cfg.be.fit_interval):
        return
    model, params, y_mean, y_std = fit_enn(state["zs"], state["ys"], cfg.be.enn_k)
    state["model"] = model
    state["params"] = params
    state["y_mean"] = y_mean
    state["y_std"] = y_std
    state["new_since_fit"] = 0
    state["phase_since_fit"] = 0


def new_be_state() -> dict:
    return {
        "zs": [],
        "ys": [],
        "model": None,
        "params": None,
        "y_mean": 0.0,
        "y_std": 1.0,
        "new_since_fit": 0,
        "phase_since_fit": 0,
    }


def predict_real_ucb(state: dict, embeddings: np.ndarray) -> np.ndarray:
    mu_std, se_std = predict_enn(state["model"], state["params"], embeddings)
    return (state["y_mean"] + state["y_std"] * mu_std) + abs(state["y_std"]) * se_std


def sample_objective_noise(objective: Any, cfg: Any, seed: int) -> np.ndarray:
    return objective.sample_noise(
        seed=int(seed),
        num_dim_target=cfg.num_dim_target,
        num_module_target=cfg.num_module_target,
    )
