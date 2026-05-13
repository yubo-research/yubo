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
    predict_enn,
)
from problems.uhd_obj import UHDVectorObjective


@dataclass
class _BSZOState:
    x: np.ndarray
    best_x: np.ndarray
    best_x_real: np.ndarray
    y_best: float | None
    y_best_real: float | None
    y_best_pred: float | None
    eval_seed: int
    next_perturb_base: int
    be_state: dict
    last_mu: float
    last_se: float
    last_imputed: bool
    num_imputed_step: int


def _run_bszo(objective: UHDVectorObjective, cfg: UHDConfig) -> None:
    from optimizer.uhd_bszo import _KalmanFilter

    use_be = cfg.optimizer == "bszo_be"
    imputer = JAXPointImputer(cfg.enn) if cfg.enn.minus_impute else None
    state = _new_bszo_state(objective)
    adapter = StepSizeAdapter(dim=objective.dim, sigma_0=cfg.bszo_epsilon)
    t0 = time.perf_counter()

    for step in range(int(cfg.num_rounds)):
        epsilon = float(adapter.sigma)
        y_best_before = state.y_best
        mu0, z_base = _bszo_eval_base(objective, state, use_be=use_be, imputer=imputer)
        perturb_base = _bszo_perturb_base(objective, cfg, state, use_be=use_be, epsilon=epsilon)
        kf = _KalmanFilter(
            int(cfg.bszo_k),
            float(cfg.bszo_sigma_p_sq),
            float(cfg.bszo_sigma_e_sq),
            float(cfg.bszo_alpha),
        )
        kf.init_step()
        noises = _bszo_eval_directions(
            objective,
            cfg,
            state,
            kf,
            perturb_base,
            mu0,
            z_base,
            epsilon,
            use_be=use_be,
            imputer=imputer,
        )
        kf.adaptive_step()
        _bszo_apply_update(cfg, state, kf, noises)
        improved = y_best_before is not None and state.y_best is not None and float(state.y_best) > float(y_best_before)
        adapter.update(accepted=improved)
        state.eval_seed += 1
        if use_be:
            fit_if_due(state.be_state, cfg)

        if _should_log(step, cfg.num_rounds, cfg.log_interval):
            source = " source = enn" if state.last_imputed else ""
            extra = f" imputed_step = {state.num_imputed_step}" if imputer is not None else ""
            best_source = _format_source_best_suffix(state, imputer is not None)
            print(
                f"EVAL: step = {step} epsilon = {adapter.sigma:.6f} "
                f"mu = {state.last_mu:.4f} se = {state.last_se:.4f} "
                f"y_best = {_format_y(state.y_best)}{best_source}{source}{extra}"
            )
            if imputer is not None:
                print(format_enn_stats(imputer, label="imputed_eval"))
        if cfg.target_accuracy is not None and state.y_best is not None and float(state.y_best) >= float(cfg.target_accuracy):
            print(f"BSZO-Vector: target reached {float(state.y_best):.4f} >= {float(cfg.target_accuracy):.4f} at step={step}")
            break

    elapsed = time.perf_counter() - t0
    _ = objective.make_policy(state.best_x)
    print(f"BSZO-Vector: elapsed = {elapsed:.2f}s ({min(step + 1, int(cfg.num_rounds))} steps)")


def _new_bszo_state(objective: UHDVectorObjective) -> _BSZOState:
    x = objective.x0
    return _BSZOState(
        x=x,
        best_x=x.copy(),
        best_x_real=x.copy(),
        y_best=None,
        y_best_real=None,
        y_best_pred=None,
        eval_seed=0,
        next_perturb_base=0,
        be_state=new_be_state(),
        last_mu=float("nan"),
        last_se=float("nan"),
        last_imputed=False,
        num_imputed_step=0,
    )


def _bszo_eval_base(
    objective: UHDVectorObjective,
    state: _BSZOState,
    *,
    use_be: bool,
    imputer: JAXPointImputer | None,
) -> tuple[float, np.ndarray | None]:
    mu0, se0 = objective.evaluate(state.x, seed=state.eval_seed)

    # Print logs if available in the side-channel
    logs = getattr(objective, "last_logs", [])
    for log in logs:
        print(log)

    state.last_mu, state.last_se = float(mu0), float(se0)
    state.last_imputed = False
    state.num_imputed_step = 0
    z_base = objective.embed(state.x) if use_be or imputer is not None else None
    _track_bszo_best(state, state.x, float(mu0))
    if imputer is not None:
        assert z_base is not None
        imputer.tell_base(z_base=np.asarray(z_base, dtype=np.float64), mu0=float(mu0))
    return float(mu0), z_base


def _bszo_perturb_base(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _BSZOState,
    *,
    use_be: bool,
    epsilon: float,
) -> int:
    if use_be and state.be_state["params"] is not None and len(state.be_state["zs"]) >= int(cfg.be.warmup):
        base, state.next_perturb_base = _select_bszo_be_base(
            objective,
            cfg,
            state.be_state,
            state.x,
            state.next_perturb_base,
            epsilon,
        )
        return base
    base = state.next_perturb_base
    state.next_perturb_base += int(cfg.bszo_k)
    return base


def _bszo_eval_directions(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _BSZOState,
    kf,
    perturb_base: int,
    mu0: float,
    z_base: np.ndarray | None,
    epsilon: float,
    *,
    use_be: bool,
    imputer: JAXPointImputer | None,
) -> list[np.ndarray]:
    noises = []
    for j in range(int(cfg.bszo_k)):
        noise = _bszo_eval_direction(
            objective,
            cfg,
            state,
            kf,
            int(perturb_base) + j,
            j,
            mu0,
            z_base,
            epsilon,
            use_be=use_be,
            imputer=imputer,
        )
        noises.append(noise)
    return noises


def _bszo_eval_direction(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: _BSZOState,
    kf,
    direction_seed: int,
    index: int,
    mu0: float,
    z_base: np.ndarray | None,
    epsilon: float,
    *,
    use_be: bool,
    imputer: JAXPointImputer | None,
) -> np.ndarray:
    noise = _noise(objective, cfg, int(direction_seed), x=state.x)
    x_eval = state.x + float(epsilon) * noise
    z_eval = objective.embed(x_eval) if use_be or imputer is not None else None
    imputed, mu, se = _bszo_eval_or_impute(objective, state, imputer, x_eval, z_eval, z_base, mu0, epsilon)
    state.last_imputed = imputed
    state.num_imputed_step += int(imputed)
    state.last_mu, state.last_se = float(mu), float(se)
    _track_bszo_best(state, x_eval, float(mu))
    y_i = (float(mu) - float(mu0)) / float(epsilon)
    if use_be and not imputed:
        _record_be(state.be_state, z_eval, y_i)
    kf.Y[index] = y_i
    kf.update_coord(index, y_i)
    kf.last_d_idx = index
    kf.last_y = y_i
    return noise


def _bszo_eval_or_impute(
    objective: UHDVectorObjective,
    state: _BSZOState,
    imputer: JAXPointImputer | None,
    x_eval: np.ndarray,
    z_eval: np.ndarray | None,
    z_base: np.ndarray | None,
    mu0: float,
    epsilon: float,
) -> tuple[bool, float, float]:
    if imputer is None:
        mu, se = objective.evaluate(x_eval, seed=state.eval_seed)
        return False, float(mu), float(se)
    assert z_eval is not None and z_base is not None
    imputed, mu, se = imputer.try_impute_eval(
        z_eval=np.asarray(z_eval, dtype=np.float64),
        z_base=np.asarray(z_base, dtype=np.float64),
        mu0=float(mu0),
        epsilon=epsilon,
    )
    if imputed:
        return True, float(mu), float(se)
    mu, se = objective.evaluate(x_eval, seed=state.eval_seed)
    imputer.calibrate_eval(
        z_eval=np.asarray(z_eval, dtype=np.float64),
        z_base=np.asarray(z_base, dtype=np.float64),
        mu_eval_real=float(mu),
        mu0=float(mu0),
        epsilon=epsilon,
    )
    imputer.tell_real_eval(
        z_eval=np.asarray(z_eval, dtype=np.float64),
        z_base=np.asarray(z_base, dtype=np.float64),
        mu_eval=float(mu),
        mu0=float(mu0),
        epsilon=epsilon,
    )
    return False, float(mu), float(se)


def _track_bszo_best(state: _BSZOState, x_eval: np.ndarray, mu: float) -> None:
    _track_legacy_best(state, x_eval, float(mu))
    _track_source_best(state, x_eval, float(mu), imputed=state.last_imputed)


def _bszo_apply_update(cfg: UHDConfig, state: _BSZOState, kf, noises: list[np.ndarray]) -> None:
    for j, noise in enumerate(noises):
        scale = float(cfg.lr) * float(kf.mu_post[j])
        if abs(scale) > 1e-30:
            state.x = state.x + scale * noise


def _select_bszo_be_base(
    objective: UHDVectorObjective,
    cfg: UHDConfig,
    state: dict,
    x: np.ndarray,
    base: int,
    epsilon: float,
) -> tuple[int, int]:
    k = int(cfg.bszo_k)
    bases = [int(base) + j * k for j in range(int(cfg.be.num_candidates))]
    candidates = np.stack([x + float(epsilon) * _noise(objective, cfg, b, x=x) for b in bases])
    embeddings = objective.embed_many(candidates)
    mu_pred, se_pred = predict_enn(state["model"], state["params"], embeddings)
    mu_real = state["y_mean"] + state["y_std"] * mu_pred
    se_real = abs(state["y_std"]) * se_pred
    best = int(np.argmax(np.abs(mu_real) + se_real))
    return bases[best], int(base) + int(cfg.be.num_candidates) * k
