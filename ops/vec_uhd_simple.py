from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ops.uhd_config import UHDConfig
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
    JAXPointImputer,
    fit_if_due,
    format_enn_stats,
    new_be_state,
    predict_real_ucb,
)
from problems.uhd_obj_types import UHDVectorObjective


@dataclass
class _SimpleState:
    x: np.ndarray
    best_x: np.ndarray
    best_x_real: np.ndarray
    y_best: float | None
    y_best_real: float | None
    y_best_pred: float | None
    next_seed: int
    be_state: dict
    current_mu_real: float | None
    current_z: np.ndarray | None
    last_mu: float
    last_se: float
    last_imputed: bool


def _run_simple(objective: UHDVectorObjective, cfg: UHDConfig) -> None:
    use_be = cfg.optimizer == "simple_be"
    imputer = JAXPointImputer(cfg.enn) if cfg.enn.minus_impute else None
    adapter = StepSizeAdapter(dim=objective.dim, sigma_0=cfg.sigma)
    state = _new_simple_state(objective)
    t0 = time.perf_counter()

    for i in range(int(cfg.num_rounds)):
        _ensure_simple_real_base(objective, state, imputer)
        if use_be and state.be_state["params"] is not None and len(state.be_state["zs"]) >= int(cfg.be.warmup):
            base = state.next_seed
            sigmas = _sample_sigmas(adapter, cfg.be.sigma_range, seed=base, n=cfg.be.num_candidates)
            seeds = [base + j for j in range(int(cfg.be.num_candidates))]
            candidates = np.stack([state.x + float(sigmas[j]) * _noise(objective, cfg, seeds[j], x=state.x) for j in range(len(seeds))])
            embeddings = objective.embed_many(candidates)
            ucb = predict_real_ucb(state.be_state, embeddings)
            best = int(np.argmax(ucb))
            seed = seeds[best]
            sigma = float(sigmas[best])
            x_candidate = candidates[best]
            z_current = embeddings[best]
            state.next_seed += int(cfg.be.num_candidates)
        else:
            seed = state.next_seed
            sigma = float(_sample_sigmas(adapter, cfg.be.sigma_range, seed=seed, n=1)[0])
            x_candidate = state.x + sigma * _noise(objective, cfg, seed, x=state.x)
            z_current = objective.embed(x_candidate) if use_be or imputer is not None else None
            state.next_seed += 1

        state.last_imputed, state.last_mu, state.last_se = _simple_eval_or_impute(
            objective,
            state,
            imputer,
            x_candidate,
            z_current,
            sigma,
            seed,
        )
        if use_be and not state.last_imputed:
            _record_be(state.be_state, z_current, state.last_mu)

        accepted = state.y_best is None or float(state.last_mu) > float(state.y_best)
        if accepted:
            state.x = x_candidate.copy()
            state.current_z = None if z_current is None else np.asarray(z_current, dtype=np.float64)
            state.current_mu_real = None if state.last_imputed else float(state.last_mu)
        adapter.update(accepted=accepted)
        if use_be:
            fit_if_due(state.be_state, cfg)
        _track_simple_best(state, x_candidate)

        if _should_log(i, cfg.num_rounds, cfg.log_interval):
            source = " source = enn" if state.last_imputed else ""
            best_source = _format_source_best_suffix(state, imputer is not None)
            print(
                f"EVAL: i_iter = {i} sigma = {adapter.sigma:.6f} "
                f"mu = {state.last_mu:.4f} se = {state.last_se:.4f} "
                f"y_best = {_format_y(state.y_best)}{best_source}{source}"
            )
            if imputer is not None:
                print(format_enn_stats(imputer, label="imputed_eval"))
        if cfg.target_accuracy is not None and float(state.last_mu) >= float(cfg.target_accuracy):
            print(f"UHD-Vector: target reached {state.last_mu:.4f} >= {float(cfg.target_accuracy):.4f} at i_iter={i}")
            break

    elapsed = time.perf_counter() - t0
    _ = objective.make_policy(state.best_x)
    print(f"UHD-Vector: elapsed = {elapsed:.2f}s ({min(i + 1, int(cfg.num_rounds))} iterations)")


def _new_simple_state(objective: UHDVectorObjective) -> _SimpleState:
    x = objective.x0
    return _SimpleState(
        x=x,
        best_x=x.copy(),
        best_x_real=x.copy(),
        y_best=None,
        y_best_real=None,
        y_best_pred=None,
        next_seed=0,
        be_state=new_be_state(),
        current_mu_real=None,
        current_z=None,
        last_mu=float("nan"),
        last_se=float("nan"),
        last_imputed=False,
    )


def _ensure_simple_real_base(
    objective: UHDVectorObjective,
    state: _SimpleState,
    imputer: JAXPointImputer | None,
) -> None:
    if imputer is None:
        return
    if state.current_z is None:
        state.current_z = objective.embed(state.x)
    if state.current_mu_real is not None:
        return
    mu0, _se0 = objective.evaluate(state.x, seed=state.next_seed)
    state.current_mu_real = float(mu0)
    _track_legacy_best(state, state.x, float(mu0))
    _track_source_best(state, state.x, float(mu0), imputed=False)
    imputer.tell_base(z_base=np.asarray(state.current_z, dtype=np.float64), mu0=float(mu0))


def _simple_eval_or_impute(
    objective: UHDVectorObjective,
    state: _SimpleState,
    imputer: JAXPointImputer | None,
    x_candidate: np.ndarray,
    z_candidate: np.ndarray | None,
    sigma: float,
    seed: int,
) -> tuple[bool, float, float]:
    if imputer is None:
        mu, se = objective.evaluate(x_candidate, seed=seed)
        return False, float(mu), float(se)

    assert z_candidate is not None and state.current_z is not None and state.current_mu_real is not None
    imputed, mu, se = imputer.try_impute_eval(
        z_eval=np.asarray(z_candidate, dtype=np.float64),
        z_base=np.asarray(state.current_z, dtype=np.float64),
        mu0=float(state.current_mu_real),
        epsilon=float(sigma),
    )
    if imputed:
        return True, float(mu), float(se)

    mu, se = objective.evaluate(x_candidate, seed=seed)
    imputer.calibrate_eval(
        z_eval=np.asarray(z_candidate, dtype=np.float64),
        z_base=np.asarray(state.current_z, dtype=np.float64),
        mu_eval_real=float(mu),
        mu0=float(state.current_mu_real),
        epsilon=float(sigma),
    )
    imputer.tell_real_eval(
        z_eval=np.asarray(z_candidate, dtype=np.float64),
        z_base=np.asarray(state.current_z, dtype=np.float64),
        mu_eval=float(mu),
        mu0=float(state.current_mu_real),
        epsilon=float(sigma),
    )
    return False, float(mu), float(se)


def _track_simple_best(state: _SimpleState, x_eval: np.ndarray) -> None:
    _track_legacy_best(state, x_eval, float(state.last_mu))
    _track_source_best(state, x_eval, float(state.last_mu), imputed=state.last_imputed)
