from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ops.uhd_config import UHDConfig
from optimizer.uhd_enn import (
    JAXMinusImputer,
    JAXPointImputer,
    fit_if_due,
    format_enn_stats,
    new_be_state,
    predict_enn,
    predict_real_ucb,
)
from optimizer.step_size_adapter import StepSizeAdapter
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
        f"steps_per_episode = {objective.steps_per_episode} eval_episodes = {objective.eval_episodes}"
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


def _should_log(i: int, n: int, interval: int) -> bool:
    return i == 0 or i == n - 1 or int(interval) <= 1 or i % int(interval) == 0


def _format_y(y: float | None) -> str:
    return "N/A" if y is None else f"{float(y):.4f}"


def _sample_sigmas(adapter: StepSizeAdapter, sigma_range: tuple[float, float] | None, *, seed: int, n: int) -> np.ndarray:
    if sigma_range is None:
        return np.full(int(n), adapter.sigma, dtype=np.float64)
    lo, hi = np.log(float(sigma_range[0])), np.log(float(sigma_range[1]))
    rng = np.random.default_rng(int(seed))
    return np.exp(rng.uniform(lo, hi, size=int(n))).astype(np.float64)


def _noise(objective, cfg: UHDConfig, seed: int, *, x: np.ndarray | None = None) -> np.ndarray:
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


def _record_be(state: dict, z: np.ndarray | None, y: float) -> None:
    assert z is not None
    state["zs"].append(np.asarray(z, dtype=np.float64))
    state["ys"].append(float(y))
    state["new_since_fit"] += 1
    state["phase_since_fit"] += 1


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
        kf = _KalmanFilter(int(cfg.bszo_k), float(cfg.bszo_sigma_p_sq), float(cfg.bszo_sigma_e_sq), float(cfg.bszo_alpha))
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


def _track_legacy_best(state: _MezoState | _BSZOState, x_eval: np.ndarray, mu: float) -> None:
    if state.y_best is not None and float(mu) <= float(state.y_best):
        return
    state.y_best = float(mu)
    state.best_x = x_eval.copy()


def _track_source_best(state: _MezoState | _BSZOState, x_eval: np.ndarray, mu: float, *, imputed: bool) -> None:
    if imputed:
        if state.y_best_pred is None or float(mu) > float(state.y_best_pred):
            state.y_best_pred = float(mu)
        return
    if state.y_best_real is not None and float(mu) <= float(state.y_best_real):
        return
    state.y_best_real = float(mu)
    state.best_x_real = x_eval.copy()


def _format_source_best_suffix(state: _MezoState | _BSZOState, enabled: bool) -> str:
    if not enabled:
        return ""
    return f" y_best_real = {_format_y(state.y_best_real)} y_best_pred = {_format_y(state.y_best_pred)}"


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
