from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .bayesian_mars_fit import _fit_bayesian_mars, _FittedBayesianMarsModel
from .mars_basis import (
    _build_main_basis,
    _MarsTerm,
    _ridge_solve,
    _safe_solve,
    _score_columns,
    _screen_features,
    _select_top,
    _standardize_y,
    _standardized_y_var,
)
from .mars_config import BayesianMarsSurrogateConfig
from .mars_fit import _fit_single_mars


@dataclass(frozen=True)
class _BayesianMarsMCMCResult:
    models: tuple[_FittedBayesianMarsModel, ...]
    weights: np.ndarray


def _fit_bayesian_mars_mcmc(
    x: np.ndarray,
    y: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    *,
    rng: np.random.Generator,
    y_var: np.ndarray | None = None,
) -> _BayesianMarsMCMCResult:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_std, _, y_scale = _standardize_y(y_arr)
    candidate_terms, candidate_cols = _build_mcmc_basis_pool(x_arr, y_std, cfg)
    if not candidate_terms:
        return _single_model_result(x_arr, y_arr, cfg, y_var)
    ctx = _MCMCContext.from_inputs(x_arr, y_arr, y_std, y_scale, cfg, y_var, len(candidate_terms))
    states, samples = _sample_basis_states(ctx, candidate_cols, rng)
    weights = _basis_state_weights(samples, states)
    models = _fit_selected_models(x_arr, y_arr, candidate_terms, states, ctx.fixed_cfg, y_var)
    return _BayesianMarsMCMCResult(models=models, weights=weights)


@dataclass(frozen=True)
class _MCMCContext:
    max_terms: int
    pool_size: int
    fixed_cfg: BayesianMarsSurrogateConfig
    noise_variance: float
    term_logit: float
    y_std: np.ndarray

    @classmethod
    def from_inputs(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        y_std: np.ndarray,
        y_scale: float,
        cfg: BayesianMarsSurrogateConfig,
        y_var: np.ndarray | None,
        pool_size: int,
    ) -> _MCMCContext:
        max_terms = max(1, min(int(cfg.basis.max_terms) - 1, max(1, int(x.shape[0]) - 1), pool_size))
        noise_variance = _estimate_bmars_noise_variance(x, y, y_std, cfg, y_var=y_var, y_scale=y_scale)
        prior_inclusion = _prior_inclusion(cfg, max_terms=max_terms, pool_size=pool_size)
        return cls(
            max_terms=max_terms,
            pool_size=pool_size,
            fixed_cfg=replace(cfg, noise_variance=noise_variance),
            noise_variance=noise_variance,
            term_logit=float(np.log(prior_inclusion) - np.log1p(-prior_inclusion)),
            y_std=y_std,
        )


def _single_model_result(
    x: np.ndarray,
    y: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    y_var: np.ndarray | None,
) -> _BayesianMarsMCMCResult:
    model = _fit_bayesian_mars(x, y, cfg, terms=(), y_var=y_var)
    return _BayesianMarsMCMCResult(models=(model,), weights=np.ones((1,), dtype=float))


def _build_mcmc_basis_pool(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
) -> tuple[tuple[_MarsTerm, ...], np.ndarray]:
    basis_cfg = cfg.basis
    num_obs = int(x.shape[0])
    if num_obs < 2 or int(basis_cfg.max_terms) <= 1:
        return (), np.zeros((num_obs, 0), dtype=float)
    pool_size = _resolved_pool_size(cfg, num_obs)
    main_terms, main_cols = _candidate_main_terms(x, y_std, cfg)
    if not main_terms:
        return (), np.zeros((num_obs, 0), dtype=float)
    terms, cols = _mcmc_main_terms(main_terms, main_cols, y_std, cfg, pool_size)
    terms, cols = _mcmc_interactions(terms, cols, main_terms, main_cols, y_std, cfg, pool_size)
    if not terms:
        return (), np.zeros((num_obs, 0), dtype=float)
    return tuple(terms), np.column_stack(cols)


def _resolved_pool_size(cfg: BayesianMarsSurrogateConfig, num_obs: int) -> int:
    max_non_intercept = max(1, min(int(cfg.basis.max_terms) - 1, max(1, int(num_obs) - 1)))
    pool_size = cfg.mcmc_pool_size
    if pool_size is None:
        pool_size = max(max_non_intercept * 8, max_non_intercept + 8)
    return max(1, int(pool_size))


def _candidate_main_terms(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    features = _screen_features(x, y_std, int(cfg.basis.feature_screen))
    quantiles = np.linspace(0.2, 0.8, int(cfg.basis.knots_per_feature))
    return _build_main_basis(x, features, quantiles)


def _mcmc_main_terms(
    main_terms: list[_MarsTerm],
    main_cols: list[np.ndarray],
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    pool_size: int,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    main_mat = np.column_stack(main_cols)
    main_scores = _score_columns(main_mat, y_std)
    main_budget = _main_pool_budget(cfg, main_terms, pool_size)
    main_idx = _select_top(main_scores, main_budget)
    terms = [main_terms[int(i)] for i in main_idx]
    cols = [main_mat[:, int(i)] for i in main_idx]
    return terms, cols


def _main_pool_budget(
    cfg: BayesianMarsSurrogateConfig,
    main_terms: list[_MarsTerm],
    pool_size: int,
) -> int:
    if int(cfg.basis.interaction_order) == 1:
        return min(len(main_terms), pool_size)
    return min(len(main_terms), max(1, pool_size // 2))


def _mcmc_interactions(
    terms: list[_MarsTerm],
    cols: list[np.ndarray],
    main_terms: list[_MarsTerm],
    main_cols: list[np.ndarray],
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    pool_size: int,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    if int(cfg.basis.interaction_order) == 1:
        return terms, cols
    inter_budget = max(0, pool_size - len(terms))
    if inter_budget <= 0 or len(main_terms) < 2:
        return terms, cols
    inter_terms, inter_cols = _mcmc_interaction_candidates(main_terms, main_cols, y_std, cfg, pool_size)
    if not inter_terms:
        return terms, cols
    inter_mat = np.column_stack(inter_cols)
    inter_idx = _select_top(_score_columns(inter_mat, y_std), inter_budget)
    terms.extend(inter_terms[int(i)] for i in inter_idx)
    cols.extend(inter_mat[:, int(i)] for i in inter_idx)
    return terms, cols


def _mcmc_interaction_candidates(
    main_terms: list[_MarsTerm],
    main_cols: list[np.ndarray],
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    pool_size: int,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    main_scores = _score_columns(np.column_stack(main_cols), y_std)
    seed_budget = min(len(main_terms), max(4, min(pool_size, max(1, int(cfg.basis.max_terms) - 1) * 4)))
    seed_idx = _select_top(main_scores, seed_budget)
    seed_terms = [main_terms[int(i)] for i in seed_idx]
    seed_cols = [main_cols[int(i)] for i in seed_idx]
    inter_terms: list[_MarsTerm] = []
    inter_cols: list[np.ndarray] = []
    for i in range(len(seed_terms)):
        feat_i = set(seed_terms[i].features)
        for j in range(i + 1, len(seed_terms)):
            if not feat_i.intersection(seed_terms[j].features):
                inter_terms.append(_MarsTerm(tuple(seed_terms[i].factors + seed_terms[j].factors)))
                inter_cols.append(seed_cols[i] * seed_cols[j])
    return inter_terms, inter_cols


def _estimate_bmars_noise_variance(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    *,
    y_var: np.ndarray | None = None,
    y_scale: float = 1.0,
) -> float:
    y_var_std = _standardized_y_var(y_var, y_scale)
    if y_var_std is not None:
        noise_variance = float(np.mean(y_var_std))
    elif cfg.noise_variance is not None:
        noise_variance = float(cfg.noise_variance)
    else:
        noise_variance = _mars_residual_noise(x, y, y_std, cfg)
    if not np.isfinite(noise_variance):
        noise_variance = float(cfg.min_noise_variance)
    return max(float(noise_variance), float(cfg.min_noise_variance))


def _mars_residual_noise(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
) -> float:
    basis_model = _fit_single_mars(x, y, cfg.basis)
    phi = basis_model._design(x)
    coef = _ridge_solve(phi, y_std, float(cfg.basis.ridge))
    residual = y_std - phi @ coef
    dof = max(int(x.shape[0]) - int(phi.shape[1]), 1)
    return float(np.sum(residual**2) / float(dof))


def _bayesian_linear_log_marginal(
    phi: np.ndarray,
    y_std: np.ndarray,
    *,
    prior_precision: float,
    intercept_prior_precision: float,
    noise_variance: float,
) -> float:
    phi_arr = np.asarray(phi, dtype=float)
    y_arr = np.asarray(y_std, dtype=float).reshape(-1)
    prior_diag = _prior_diag(phi_arr.shape[1], prior_precision, intercept_prior_precision)
    precision = np.diag(prior_diag) + (phi_arr.T @ phi_arr) / max(float(noise_variance), 1e-12)
    sign, logdet_precision = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(logdet_precision):
        return -np.inf
    rhs = (phi_arr.T @ y_arr) / max(float(noise_variance), 1e-12)
    sol = _safe_solve(precision, rhs)
    quad = float((y_arr @ y_arr) / max(float(noise_variance), 1e-12) - rhs @ sol)
    return _log_marginal_value(phi_arr.shape[0], prior_diag, logdet_precision, quad, noise_variance)


def _prior_diag(p: int, prior_precision: float, intercept_prior_precision: float) -> np.ndarray:
    prior_diag = float(prior_precision) * np.ones((int(p),), dtype=float)
    prior_diag[0] = max(float(intercept_prior_precision), 1e-12)
    return prior_diag


def _log_marginal_value(
    n: int,
    prior_diag: np.ndarray,
    logdet_precision: float,
    quad: float,
    noise_variance: float,
) -> float:
    noise = max(float(noise_variance), 1e-12)
    return float(-0.5 * int(n) * np.log(2.0 * np.pi * noise) + 0.5 * (float(np.sum(np.log(prior_diag))) - logdet_precision) - 0.5 * quad)


def _mcmc_move_probs(k: int, *, max_terms: int, pool_size: int) -> tuple[float, float]:
    if pool_size <= 0 or max_terms <= 0:
        return 0.0, 0.0
    if k <= 0:
        return 1.0, 0.0
    if k >= min(max_terms, pool_size):
        return 0.0, 1.0
    return 0.5, 0.5


def _propose_mcmc_basis_state(
    state: tuple[int, ...],
    *,
    pool_size: int,
    max_terms: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], float, float]:
    selected = set(int(i) for i in state)
    p_add, p_drop = _mcmc_move_probs(len(selected), max_terms=max_terms, pool_size=pool_size)
    if p_add <= 0.0 and p_drop <= 0.0:
        return state, 0.0, 0.0
    if p_add > 0.0 and (p_drop <= 0.0 or rng.random() < p_add):
        return _propose_add(state, selected, p_add, max_terms, pool_size, rng)
    return _propose_drop(state, p_drop, max_terms, pool_size, rng)


def _propose_add(
    state: tuple[int, ...],
    selected: set[int],
    p_add: float,
    max_terms: int,
    pool_size: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], float, float]:
    choices = [i for i in range(pool_size) if i not in selected]
    if not choices:
        return state, 0.0, 0.0
    add_idx = int(choices[int(rng.integers(0, len(choices)))])
    new_state = tuple(sorted((*state, add_idx)))
    _, reverse_drop = _mcmc_move_probs(len(new_state), max_terms=max_terms, pool_size=pool_size)
    return new_state, float(np.log(p_add / float(len(choices)))), float(np.log(reverse_drop / float(len(new_state))))


def _propose_drop(
    state: tuple[int, ...],
    p_drop: float,
    max_terms: int,
    pool_size: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], float, float]:
    if not state:
        return state, 0.0, 0.0
    drop_idx = int(state[int(rng.integers(0, len(state)))])
    new_state = tuple(i for i in state if int(i) != drop_idx)
    reverse_add, _ = _mcmc_move_probs(len(new_state), max_terms=max_terms, pool_size=pool_size)
    q_reverse = reverse_add / float(max(1, pool_size - len(new_state)))
    return new_state, float(np.log(p_drop / float(len(state)))), float(np.log(q_reverse))


def _sample_basis_states(
    ctx: _MCMCContext,
    candidate_cols: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    state = _initial_state(candidate_cols, ctx)
    log_target = _LogTarget(ctx, candidate_cols)
    state_log_target = log_target(state)
    samples: list[tuple[int, ...]] = []
    best_state, best_log_target = state, state_log_target
    for step in range(int(ctx.fixed_cfg.mcmc_burn_in) + int(ctx.fixed_cfg.mcmc_steps)):
        state, state_log_target = _mcmc_step(state, state_log_target, ctx, log_target, rng)
        if state_log_target > best_log_target:
            best_state, best_log_target = state, state_log_target
        if _keep_mcmc_step(step, ctx.fixed_cfg):
            samples.append(state)
    if not samples:
        samples = [best_state]
    states = _selected_states(samples, best_state, int(ctx.fixed_cfg.mcmc_num_models))
    return states, samples


class _LogTarget:
    def __init__(self, ctx: _MCMCContext, candidate_cols: np.ndarray) -> None:
        self._ctx = ctx
        self._candidate_cols = candidate_cols
        self._ones = np.ones((candidate_cols.shape[0], 1), dtype=float)
        self._cache: dict[tuple[int, ...], float] = {}

    def __call__(self, state: tuple[int, ...]) -> float:
        key = tuple(sorted(int(i) for i in state))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._compute(key)
        self._cache[key] = value
        return value

    def _compute(self, state: tuple[int, ...]) -> float:
        phi = np.column_stack([self._ones, self._candidate_cols[:, list(state)]]) if state else self._ones
        log_ml = _bayesian_linear_log_marginal(
            phi,
            self._ctx.y_std,
            prior_precision=float(self._ctx.fixed_cfg.prior_precision),
            intercept_prior_precision=float(self._ctx.fixed_cfg.intercept_prior_precision),
            noise_variance=float(self._ctx.noise_variance),
        )
        return float(log_ml + len(state) * self._ctx.term_logit)


def _initial_state(candidate_cols: np.ndarray, ctx: _MCMCContext) -> tuple[int, ...]:
    scores = _score_columns(candidate_cols, ctx.y_std)
    init_size = min(ctx.max_terms, max(1, ctx.max_terms // 2), ctx.pool_size)
    return tuple(sorted(int(i) for i in _select_top(scores, init_size)))


def _mcmc_step(
    state: tuple[int, ...],
    state_log_target: float,
    ctx: _MCMCContext,
    log_target: _LogTarget,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], float]:
    proposal, log_q_forward, log_q_reverse = _propose_mcmc_basis_state(
        state,
        pool_size=ctx.pool_size,
        max_terms=ctx.max_terms,
        rng=rng,
    )
    proposal_log_target = log_target(proposal)
    log_alpha = proposal_log_target - state_log_target + log_q_reverse - log_q_forward
    if np.log(max(float(rng.random()), 1e-300)) < min(0.0, float(log_alpha)):
        return proposal, proposal_log_target
    return state, state_log_target


def _keep_mcmc_step(step: int, cfg: BayesianMarsSurrogateConfig) -> bool:
    if step < int(cfg.mcmc_burn_in):
        return False
    return (step - int(cfg.mcmc_burn_in)) % int(cfg.mcmc_thin) == 0


def _selected_states(
    samples: list[tuple[int, ...]],
    best_state: tuple[int, ...],
    limit: int,
) -> list[tuple[int, ...]]:
    counts = Counter(samples)
    selected = [state for state, _ in counts.most_common(int(limit))]
    if best_state not in selected:
        selected[-1] = best_state
    return selected


def _basis_state_weights(
    samples: list[tuple[int, ...]],
    selected: list[tuple[int, ...]],
) -> np.ndarray:
    counts = Counter(samples)
    raw = np.asarray([counts.get(state, 0) for state in selected], dtype=float)
    raw[raw <= 0.0] = 1.0
    return raw / float(np.sum(raw))


def _fit_selected_models(
    x: np.ndarray,
    y: np.ndarray,
    candidate_terms: tuple[_MarsTerm, ...],
    states: list[tuple[int, ...]],
    cfg: BayesianMarsSurrogateConfig,
    y_var: np.ndarray | None,
) -> tuple[_FittedBayesianMarsModel, ...]:
    return tuple(_fit_bayesian_mars(x, y, cfg, terms=tuple(candidate_terms[int(i)] for i in state), y_var=y_var) for state in states)


def _prior_inclusion(
    cfg: BayesianMarsSurrogateConfig,
    *,
    max_terms: int,
    pool_size: int,
) -> float:
    if cfg.mcmc_term_prior is not None:
        return float(cfg.mcmc_term_prior)
    expected_terms = max(1.0, 0.5 * float(max_terms))
    return min(0.5, max(1e-4, expected_terms / float(max(pool_size, 1))))
