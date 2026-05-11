from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ops.uhd_config import UHDConfig
from ops.vec_uhd_bszo import _run_bszo
from ops.vec_uhd_common import (
    _format_source_best_suffix,
    _format_y,
    _noise,
    _record_be,
    _sample_sigmas,
    _should_log,
    _track_legacy_best,
    _track_source_best,
)
from optimizer.step_size_adapter import StepSizeAdapter
from optimizer.uhd_enn import (
    JAXMinusImputer,
    fit_if_due,
    format_enn_stats,
    new_be_state,
    predict_enn,
    predict_real_ucb,
)
from problems.uhd_obj import UHDVectorObjective


_BE_OPTIMIZERS = {"simple_be", "mezo_be", "bszo_be"}


def run_uhd_vector_loop(cfg: UHDConfig) -> None:
    optimizer = str(cfg.optimizer)
    _validate_minus_impute_optimizer(cfg)
    if cfg.enn.minus_impute and int(cfg.be.num_probes) < 1:
        raise ValueError("UHD vector enn_minus_impute requires be_num_probes >= 1 for behavior embeddings.")

    from problems.uhd_obj import build_uhd_vector_objective

    embed_num_probes = cfg.be.num_probes if optimizer in _BE_OPTIMIZERS or cfg.enn.minus_impute else 0
    built = build_uhd_vector_objective(cfg, embed_num_probes=embed_num_probes)
    objective = built.objective

    print(
        "UHD-Vector: "
        f"source = {built.source} env_tag = {cfg.env_tag} optimizer = {optimizer} dim = {objective.dim} "
        f"perturb = {cfg.perturb_backend} "
        f"steps_per_episode = {objective.steps_per_episode} num_envs = {objective.num_envs}"
    )
    try:
        if optimizer in {"simple", "simple_be"}:
            _run_simple(objective, cfg)
        elif optimizer in {"mezo", "mezo_be"}:
            _run_mezo(objective, cfg)
        elif optimizer in {"bszo", "bszo_be"}:
            _run_bszo(objective, cfg)
        else:
            raise ValueError(f"Unknown UHD vector optimizer: {optimizer}")
    finally:
        close = getattr(objective, "close", None)
        if callable(close):
            close()


def _validate_minus_impute_optimizer(cfg: UHDConfig) -> None:
    if not cfg.enn.minus_impute or str(cfg.optimizer) in {"mezo", "mezo_be", "bszo", "bszo_be"}:
        return
    raise ValueError("UHD vector enn_minus_impute is currently supported for optimizer='mezo', 'mezo_be', 'bszo', and 'bszo_be'.")


@dataclass
class _MezoState:
    x: np.ndarray
    best_x: np.ndarray
    best_x_real: np.ndarray
    y_best: float | None
    y_best_real: float | None
    y_best_pred: float | None
    seed: int
    positive_phase: bool
    step_seed: int
    step_noise: np.ndarray
    mu_plus: float
    grad_sq_ema: float
    be_state: dict
    selected_embeddings: tuple[np.ndarray, np.ndarray] | None
    last_mu: float
    last_se: float
    last_imputed: bool


def _run_simple(objective: UHDVectorObjective, cfg: UHDConfig) -> None:
    use_be = cfg.optimizer == "simple_be"
    adapter = StepSizeAdapter(dim=objective.dim, sigma_0=cfg.sigma)
    x = objective.x0
    best_x = x.copy()
    y_best: float | None = None
    next_seed = 0
    state = new_be_state()
    t0 = time.perf_counter()

    for i in range(int(cfg.num_rounds)):
        if use_be and state["params"] is not None and len(state["zs"]) >= int(cfg.be.warmup):
            base = next_seed
            sigmas = _sample_sigmas(adapter, cfg.be.sigma_range, seed=base, n=cfg.be.num_candidates)
            seeds = [base + j for j in range(int(cfg.be.num_candidates))]
            candidates = np.stack([x + float(sigmas[j]) * _noise(objective, cfg, seeds[j], x=x) for j in range(len(seeds))])
            embeddings = objective.embed_many(candidates)
            ucb = predict_real_ucb(state, embeddings)
            best = int(np.argmax(ucb))
            seed = seeds[best]
            x_candidate = candidates[best]
            z_current = embeddings[best]
            next_seed += int(cfg.be.num_candidates)
        else:
            seed = next_seed
            sigma = float(_sample_sigmas(adapter, cfg.be.sigma_range, seed=seed, n=1)[0])
            x_candidate = x + sigma * _noise(objective, cfg, seed, x=x)
            z_current = objective.embed(x_candidate) if use_be else None
            next_seed += 1

        mu, se = objective.evaluate(x_candidate, seed=seed)
        if use_be:
            state["zs"].append(np.asarray(z_current, dtype=np.float64))
            state["ys"].append(float(mu))
            state["new_since_fit"] += 1
            state["phase_since_fit"] += 1

        accepted = y_best is None or float(mu) > float(y_best)
        if accepted:
            y_best = float(mu)
            best_x = x_candidate.copy()
            x = x_candidate.copy()
        adapter.update(accepted=accepted)
        if use_be:
            fit_if_due(state, cfg)

        if _should_log(i, cfg.num_rounds, cfg.log_interval):
            print(f"EVAL: i_iter = {i} sigma = {adapter.sigma:.6f} mu = {mu:.4f} se = {se:.4f} y_best = {_format_y(y_best)}")
        if cfg.target_accuracy is not None and float(mu) >= float(cfg.target_accuracy):
            print(f"UHD-Vector: target reached {mu:.4f} >= {float(cfg.target_accuracy):.4f} at i_iter={i}")
            break

    elapsed = time.perf_counter() - t0
    _ = objective.make_policy(best_x)
    print(f"UHD-Vector: elapsed = {elapsed:.2f}s ({min(i + 1, int(cfg.num_rounds))} iterations)")


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

        _track_mezo_best(state, x_eval)
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
        best_x=x.copy(),
        best_x_real=x.copy(),
        y_best=None,
        y_best_real=None,
        y_best_pred=None,
        seed=0,
        positive_phase=True,
        step_seed=0,
        step_noise=np.zeros(objective.dim, dtype=np.float64),
        mu_plus=0.0,
        grad_sq_ema=0.0,
        be_state=new_be_state(),
        selected_embeddings=None,
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
        state.step_seed, state.selected_embeddings = _select_mezo_be_seed(objective, cfg, state.be_state, state.x, state.seed)
    elif imputer is not None:
        state.step_seed, state.selected_embeddings = imputer.choose_seed_ucb(objective=objective, cfg=cfg, x=state.x, base_seed=state.seed)
    else:
        state.step_seed = state.seed
        state.selected_embeddings = None
    state.step_noise = _noise(objective, cfg, state.step_seed, x=state.x)
    x_eval = state.x + float(cfg.sigma) * state.step_noise
    _ensure_mezo_embeddings(objective, cfg, state, x_eval, need_embeddings=use_be or imputer is not None)
    z_current = state.selected_embeddings[0] if state.selected_embeddings is not None else None
    state.last_mu, state.last_se = objective.evaluate(x_eval, seed=state.step_seed)
    state.mu_plus = float(state.last_mu)
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
    z_pair = objective.embed_many(np.stack([x_eval, x_minus]))
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
    z_current, z_plus = _mezo_negative_embeddings(objective, cfg, state, x_eval, need_embeddings=use_be or imputer is not None)
    state.last_imputed, state.last_mu, state.last_se = _mezo_minus_value(objective, cfg, state, imputer, x_eval, z_current, z_plus)
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
    imputer.calibrate_minus(z_minus=np.asarray(z_current, dtype=np.float64), z_delta=z_delta, mu_minus_real=float(mu))
    imputer.tell_real_minus(z_minus=np.asarray(z_current, dtype=np.float64), z_delta=z_delta, mu_minus=float(mu))
    return False, float(mu), float(se)


def _apply_mezo_step(cfg: UHDConfig, state: _MezoState) -> None:
    projected_grad = (state.mu_plus - float(state.last_mu)) / (2.0 * float(cfg.sigma))
    state.grad_sq_ema = 0.9 * state.grad_sq_ema + 0.1 * projected_grad**2
    rms = math.sqrt(state.grad_sq_ema) + 1e-8
    state.x = state.x + float(cfg.lr) * projected_grad * state.step_noise / rms
    state.seed = state.step_seed + 1
    state.positive_phase = True
    state.selected_embeddings = None


def _track_mezo_best(state: _MezoState, x_eval: np.ndarray) -> None:
    _track_legacy_best(state, x_eval, float(state.last_mu))
    _track_source_best(state, x_eval, float(state.last_mu), imputed=state.last_imputed)


def _select_mezo_be_seed(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: dict,
    x: np.ndarray,
    base_seed: int,
) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
    seeds = [int(base_seed) + j for j in range(int(cfg.be.num_candidates))]
    noises = [_noise(objective, cfg, s, x=x) for s in seeds]
    x_plus = np.stack([x + float(cfg.sigma) * n for n in noises])
    x_minus = np.stack([x - float(cfg.sigma) * n for n in noises])
    z_plus = objective.embed_many(x_plus)
    z_minus = objective.embed_many(x_minus)
    mu_plus, se_plus = predict_enn(state["model"], state["params"], z_plus)
    mu_minus, se_minus = predict_enn(state["model"], state["params"], z_minus)
    two_sigma = 2.0 * float(cfg.sigma)
    grad = (mu_plus - mu_minus) / two_sigma
    se_grad = np.sqrt(se_plus**2 + se_minus**2) / two_sigma
    best = int(np.argmax(np.abs(grad) + se_grad))
    return seeds[best], (z_plus[best], z_minus[best])
