from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ops.uhd_config import UHDConfig
from ops.vec_uhd_arrays import copy_vector, stack_vectors, zeros_vector
from ops.vec_uhd_be import be_pick_mezo_seed
from ops.vec_uhd_common import (
    _format_source_best_suffix,
    _format_y,
    _noise,
    _record_be,
    _should_log,
    _track_legacy_best,
    _track_source_best,
)
from optimizer.uhd_enn import (
    JAXMinusImputer,
    fit_if_due,
    format_enn_stats,
    new_be_state,
    predict_enn,
)
from problems.uhd_obj_types import UHDVectorObjective


@dataclass
class _MezoState:
    x: object
    best_x: object
    best_x_real: object
    y_best: float | None
    y_best_real: float | None
    y_best_pred: float | None
    seed: int
    positive_phase: bool
    step_seed: int
    step_noise: object
    mu_plus: float
    grad_sq_ema: float
    be_state: dict
    selected_embeddings: tuple[np.ndarray, np.ndarray] | None
    pending_minus: tuple[float, float] | None
    last_mu: float
    last_se: float
    last_imputed: bool


def _run_mezo(objective: UHDVectorObjective, cfg: UHDConfig) -> None:
    use_be = cfg.optimizer == "mezo_be"
    imputer = JAXMinusImputer(cfg.enn) if cfg.enn.minus_impute else None
    state = _new_mezo_state(objective)
    t0 = time.perf_counter()

    for i in range(int(cfg.num_rounds)):
        if state.positive_phase:
            x_eval = _mezo_positive_phase(objective, cfg, state, use_be=use_be, imputer=imputer)
        else:
            x_eval = _mezo_negative_phase(objective, cfg, state, use_be=use_be, imputer=imputer)

        _track_mezo_best(objective, state, x_eval)
        if _should_log(i, cfg.num_rounds, cfg.log_interval):
            source = " source = enn" if state.last_imputed else ""
            best_source = _format_source_best_suffix(state, imputer is not None)
            print(
                f"EVAL: i_iter = {i} sigma = {float(cfg.sigma):.6f} "
                f"mu = {state.last_mu:.4f} se = {state.last_se:.4f} "
                f"y_best = {_format_y(state.y_best)}{best_source}{source}"
            )
            if imputer is not None:
                print(format_enn_stats(imputer))
        if cfg.target_accuracy is not None and float(state.last_mu) >= float(cfg.target_accuracy):
            print(f"UHD-Vector: target reached {state.last_mu:.4f} >= {float(cfg.target_accuracy):.4f} at i_iter={i}")
            break

    elapsed = time.perf_counter() - t0
    _ = objective.make_policy(state.best_x)
    print(f"UHD-Vector: elapsed = {elapsed:.2f}s ({min(i + 1, int(cfg.num_rounds))} iterations)")


def _new_mezo_state(objective: UHDVectorObjective) -> _MezoState:
    x = objective.x0
    return _MezoState(
        x=x,
        best_x=copy_vector(objective, x),
        best_x_real=copy_vector(objective, x),
        y_best=None,
        y_best_real=None,
        y_best_pred=None,
        seed=0,
        positive_phase=True,
        step_seed=0,
        step_noise=zeros_vector(objective, objective.dim),
        mu_plus=0.0,
        grad_sq_ema=0.0,
        be_state=new_be_state(),
        selected_embeddings=None,
        pending_minus=None,
        last_mu=float("nan"),
        last_se=float("nan"),
        last_imputed=False,
    )


def _mezo_positive_phase(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _MezoState,
    *,
    use_be: bool,
    imputer: JAXMinusImputer | None,
) -> np.ndarray:
    if use_be and state.be_state["params"] is not None and len(state.be_state["zs"]) >= int(cfg.be.warmup):
        state.step_seed, state.selected_embeddings = _select_mezo_be_seed(
            objective,
            cfg,
            state.be_state,
            state.x,
            state.seed,
        )
    elif imputer is not None:
        state.step_seed, state.selected_embeddings = imputer.choose_seed_ucb(objective=objective, cfg=cfg, x=state.x, base_seed=state.seed)
    else:
        state.step_seed = state.seed
        state.selected_embeddings = None
    state.step_noise = _noise(objective, cfg, state.step_seed, x=state.x)
    x_eval = state.x + float(cfg.sigma) * state.step_noise
    _ensure_mezo_embeddings(objective, cfg, state, x_eval, need_embeddings=use_be or imputer is not None)
    z_current = state.selected_embeddings[0] if state.selected_embeddings is not None else None

    # TextObjective now stores logs in last_logs attribute
    if imputer is None and hasattr(objective, "evaluate_many_common_seed"):
        x_minus = state.x - float(cfg.sigma) * state.step_noise
        means, ses = objective.evaluate_many_common_seed(stack_vectors(objective, [x_eval, x_minus]), seed=state.step_seed)
        state.last_mu, state.last_se = float(means[0]), float(ses[0])
        state.pending_minus = (float(means[1]), float(ses[1]))
    else:
        state.last_mu, state.last_se = objective.evaluate(x_eval, seed=state.step_seed)
        state.pending_minus = None
    state.mu_plus = float(state.last_mu)

    if _should_log(state.seed, cfg.num_rounds, cfg.log_interval):
        logs = getattr(objective, "last_logs", [])
        for log in logs:
            print(log)

    if use_be:
        _record_be(state.be_state, z_current, state.last_mu)
    if imputer is not None:
        assert state.selected_embeddings is not None
        imputer.tell_plus(
            z_plus=np.asarray(state.selected_embeddings[0], dtype=np.float64),
            z_minus=np.asarray(state.selected_embeddings[1], dtype=np.float64),
            mu_plus=float(state.last_mu),
        )
    state.last_imputed = False
    state.positive_phase = False
    return x_eval


def _ensure_mezo_embeddings(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _MezoState,
    x_eval: np.ndarray,
    *,
    need_embeddings: bool,
) -> None:
    if state.selected_embeddings is not None or not need_embeddings:
        return
    x_minus = state.x - float(cfg.sigma) * state.step_noise
    z_pair = objective.embed_many(stack_vectors(objective, [x_eval, x_minus]))
    state.selected_embeddings = (z_pair[0], z_pair[1])


def _mezo_negative_phase(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _MezoState,
    *,
    use_be: bool,
    imputer: JAXMinusImputer | None,
) -> np.ndarray:
    x_eval = state.x - float(cfg.sigma) * state.step_noise
    z_current, z_plus = _mezo_negative_embeddings(
        objective,
        cfg,
        state,
        x_eval,
        need_embeddings=use_be or imputer is not None,
    )
    state.last_imputed, state.last_mu, state.last_se = _mezo_minus_value(objective, cfg, state, imputer, x_eval, z_current, z_plus)
    if _should_log(state.seed, cfg.num_rounds, cfg.log_interval):
        logs = getattr(objective, "last_logs", [])
        for log in logs:
            print(log)
    if use_be and not state.last_imputed:
        _record_be(state.be_state, z_current, state.last_mu)
    _apply_mezo_step(cfg, state)
    if use_be:
        fit_if_due(state.be_state, cfg, require_pair=True)
    return x_eval


def _mezo_negative_embeddings(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _MezoState,
    x_eval: np.ndarray,
    *,
    need_embeddings: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if state.selected_embeddings is not None:
        return state.selected_embeddings[1], state.selected_embeddings[0]
    if not need_embeddings:
        return None, None
    return objective.embed(x_eval), None


def _mezo_minus_value(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _MezoState,
    imputer: JAXMinusImputer | None,
    x_eval: np.ndarray,
    z_current: np.ndarray | None,
    z_plus: np.ndarray | None,
) -> tuple[bool, float, float]:
    if state.pending_minus is not None:
        mu, se = state.pending_minus
        state.pending_minus = None
        return False, float(mu), float(se)
    if imputer is None:
        mu, se = objective.evaluate(x_eval, seed=state.step_seed)
        return False, float(mu), float(se)
    if z_plus is None:
        z_plus = objective.embed(state.x + float(cfg.sigma) * state.step_noise)
    assert z_current is not None
    z_delta = np.asarray(z_plus, dtype=np.float64) - np.asarray(z_current, dtype=np.float64)
    imputed, mu, se = imputer.try_impute_minus(z_minus=np.asarray(z_current, dtype=np.float64), z_delta=z_delta)
    if imputed:
        return True, float(mu), float(se)
    mu, se = objective.evaluate(x_eval, seed=state.step_seed)
    imputer.calibrate_minus(
        z_minus=np.asarray(z_current, dtype=np.float64),
        z_delta=z_delta,
        mu_minus_real=float(mu),
    )
    imputer.tell_real_minus(
        z_minus=np.asarray(z_current, dtype=np.float64),
        z_delta=z_delta,
        mu_minus=float(mu),
    )
    return False, float(mu), float(se)


def _apply_mezo_step(cfg: UHDConfig, state: _MezoState) -> None:
    projected_grad = (state.mu_plus - float(state.last_mu)) / (2.0 * float(cfg.sigma))
    state.grad_sq_ema = 0.9 * state.grad_sq_ema + 0.1 * projected_grad**2
    rms = math.sqrt(state.grad_sq_ema) + 1e-8
    state.x = state.x + float(cfg.lr) * projected_grad * state.step_noise / rms
    state.seed = state.step_seed + 1
    state.positive_phase = True
    state.selected_embeddings = None


def _track_mezo_best(objective: UHDVectorObjective, state: _MezoState, x_eval: np.ndarray) -> None:
    _track_legacy_best(objective, state, x_eval, float(state.last_mu))
    _track_source_best(objective, state, x_eval, float(state.last_mu), imputed=state.last_imputed)


def _select_mezo_be_seed(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: dict,
    x: np.ndarray,
    base_seed: int,
) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
    seeds = [int(base_seed) + j for j in range(int(cfg.be.num_candidates))]
    noises = [_noise(objective, cfg, s, x=x) for s in seeds]
    x_plus = stack_vectors(objective, [x + float(cfg.sigma) * n for n in noises])
    x_minus = stack_vectors(objective, [x - float(cfg.sigma) * n for n in noises])
    sim_best = be_pick_mezo_seed(objective, x_plus, x_minus, seeds, sigma=float(cfg.sigma))
    if sim_best is not None:
        return seeds[sim_best], (
            objective.embed(x_plus[sim_best]),
            objective.embed(x_minus[sim_best]),
        )
    z_plus = objective.embed_many(x_plus)
    z_minus = objective.embed_many(x_minus)
    mu_plus, se_plus = predict_enn(state["model"], state["params"], z_plus)
    mu_minus, se_minus = predict_enn(state["model"], state["params"], z_minus)
    two_sigma = 2.0 * float(cfg.sigma)
    grad = (mu_plus - mu_minus) / two_sigma
    se_grad = np.sqrt(se_plus**2 + se_minus**2) / two_sigma
    best = int(np.argmax(np.abs(grad) + se_grad))
    return seeds[best], (z_plus[best], z_minus[best])
