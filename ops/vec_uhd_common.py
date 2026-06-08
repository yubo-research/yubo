from __future__ import annotations

from typing import Any

import numpy as np

from ops.uhd_config import UHDConfig
from ops.vec_uhd_arrays import copy_vector
from optimizer.step_size_adapter import StepSizeAdapter


def _should_log(i: int, n: int, interval: int) -> bool:
    return i < min(5, int(n)) or i == n - 1 or int(interval) <= 1 or i % int(interval) == 0


def _format_y(y: float | None) -> str:
    return "N/A" if y is None else f"{float(y):.4f}"


def _sample_sigmas(
    adapter: StepSizeAdapter,
    sigma_range: tuple[float, float] | None,
    *,
    seed: int,
    n: int,
) -> np.ndarray:
    if sigma_range is None:
        return np.full(int(n), adapter.sigma, dtype=np.float64)
    lo, hi = np.log(float(sigma_range[0])), np.log(float(sigma_range[1]))
    rng = np.random.default_rng(int(seed))
    return np.exp(rng.uniform(lo, hi, size=int(n))).astype(np.float64)


def _noise(objective: Any, cfg: UHDConfig, seed: int, *, x: Any | None = None) -> Any:
    if cfg.perturb_backend == "eggroll":
        if x is None:
            raise ValueError("EggRoll noiser perturbations require the current flat vector x.")
        if not hasattr(objective, "sample_eggroll_noiser_noise"):
            raise ValueError("perturb='eggroll' is only supported for objectives exposing EggRoll noiser materialization.")
        return objective.sample_eggroll_noiser_noise(
            x,
            seed=int(seed),
            noiser_name=cfg.eggroll_noiser,
            rank=cfg.eggroll_rank,
            group_size=cfg.eggroll_group_size,
            freeze_nonlora=cfg.eggroll_freeze_nonlora,
        )
    return objective.sample_noise(
        seed=int(seed),
        num_dim_target=cfg.num_dim_target,
        num_module_target=cfg.num_module_target,
    )


def _record_be(state: dict, z: np.ndarray | None, y: float) -> None:
    assert z is not None
    state["zs"].append(np.asarray(z, dtype=np.float64))
    state["ys"].append(float(y))
    state["new_since_fit"] += 1
    state["phase_since_fit"] += 1


def _track_legacy_best(objective: Any, state: Any, x_eval: Any, mu: float) -> None:
    if state.y_best is not None and float(mu) <= float(state.y_best):
        return
    state.y_best = float(mu)
    state.best_x = copy_vector(objective, x_eval)


def _track_source_best(objective: Any, state: Any, x_eval: Any, mu: float, *, imputed: bool) -> None:
    if imputed:
        if state.y_best_pred is None or float(mu) > float(state.y_best_pred):
            state.y_best_pred = float(mu)
        return
    if state.y_best_real is not None and float(mu) <= float(state.y_best_real):
        return
    state.y_best_real = float(mu)
    state.best_x_real = copy_vector(objective, x_eval)


def _format_source_best_suffix(state: Any, enabled: bool) -> str:
    if not enabled:
        return ""
    return f" y_best_real = {_format_y(state.y_best_real)} y_best_pred = {_format_y(state.y_best_pred)}"
