from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .bayesian_mars_fit import _fit_bayesian_mars, _FittedBayesianMarsModel
from .bayesian_mars_proposal import _TermProposal
from .mars_basis import (
    _MarsTerm,
    _ridge_solve,
    _safe_solve,
    _standardize_y,
    _standardized_y_var,
)
from .mars_config import BayesianMarsSurrogateConfig
from .mars_fit import _fit_single_mars


@dataclass(frozen=True)
class _BayesianMarsMCMCResult:
    models: tuple[_FittedBayesianMarsModel, ...]
    weights: np.ndarray


@dataclass(frozen=True)
class _MCMCContext:
    cfg: BayesianMarsSurrogateConfig
    proposal: _TermProposal
    max_terms: int
    noise_variance: float
    term_logit: float
    x: np.ndarray
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
    ) -> _MCMCContext:
        proposal = _TermProposal.from_inputs(x, y_std, cfg)
        max_terms = max(1, min(int(cfg.basis.max_terms) - 1, max(1, int(x.shape[0]) - 1)))
        noise_variance = _estimate_bmars_noise_variance(x, y, y_std, cfg, y_var=y_var, y_scale=y_scale)
        prior_inclusion = _prior_inclusion(cfg, max_terms=max_terms)
        return cls(
            cfg=replace(cfg, noise_variance=noise_variance),
            proposal=proposal,
            max_terms=max_terms,
            noise_variance=noise_variance,
            term_logit=float(np.log(prior_inclusion) - np.log1p(-prior_inclusion)),
            x=x,
            y_std=y_std,
        )


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
    ctx = _MCMCContext.from_inputs(x_arr, y_arr, y_std, y_scale, cfg, y_var)
    if not ctx.proposal.has_terms():
        return _single_model_result(x_arr, y_arr, ctx.cfg, y_var)
    states, samples = _sample_basis_states(ctx, rng)
    weights = _basis_state_weights(samples, states)
    models = _fit_selected_models(x_arr, y_arr, states, ctx.cfg, y_var)
    return _BayesianMarsMCMCResult(models=models, weights=weights)


def _single_model_result(
    x: np.ndarray,
    y: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
    y_var: np.ndarray | None,
) -> _BayesianMarsMCMCResult:
    model = _fit_bayesian_mars(x, y, cfg, terms=(), y_var=y_var)
    return _BayesianMarsMCMCResult(models=(model,), weights=np.ones((1,), dtype=float))


def _sample_basis_states(
    ctx: _MCMCContext,
    rng: np.random.Generator,
) -> tuple[list[tuple[_MarsTerm, ...]], list[tuple[_MarsTerm, ...]]]:
    state = _initial_state(ctx, rng)
    log_target = _LogTarget(ctx)
    state_log_target = log_target(state)
    samples: list[tuple[_MarsTerm, ...]] = []
    best_state, best_log_target = state, state_log_target
    for step in range(int(ctx.cfg.mcmc_burn_in) + int(ctx.cfg.mcmc_steps)):
        state, state_log_target = _mcmc_step(state, state_log_target, ctx, log_target, rng)
        if state_log_target > best_log_target:
            best_state, best_log_target = state, state_log_target
        if _keep_mcmc_step(step, ctx.cfg):
            samples.append(state)
    if not samples:
        samples = [best_state]
    return _selected_states(samples, best_state, int(ctx.cfg.mcmc_num_models)), samples


class _LogTarget:
    def __init__(self, ctx: _MCMCContext) -> None:
        self._ctx = ctx
        self._cache: dict[tuple[_MarsTerm, ...], float] = {}

    def __call__(self, state: tuple[_MarsTerm, ...]) -> float:
        key = _state_key(state)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._compute(key)
        self._cache[key] = value
        return value

    def _compute(self, state: tuple[_MarsTerm, ...]) -> float:
        phi = _design_for_state(self._ctx.x, state)
        log_ml = _bayesian_linear_log_marginal(
            phi,
            self._ctx.y_std,
            prior_precision=float(self._ctx.cfg.prior_precision),
            intercept_prior_precision=float(self._ctx.cfg.intercept_prior_precision),
            noise_variance=float(self._ctx.noise_variance),
        )
        return float(log_ml + len(state) * self._ctx.term_logit)


def _initial_state(ctx: _MCMCContext, rng: np.random.Generator) -> tuple[_MarsTerm, ...]:
    state: tuple[_MarsTerm, ...] = ()
    target = min(ctx.max_terms, max(1, ctx.max_terms // 2))
    for _ in range(target):
        term = ctx.proposal.sample_excluding(set(state), rng)
        if term is None:
            break
        state = _state_key((*state, term))
    return state


def _mcmc_step(
    state: tuple[_MarsTerm, ...],
    state_log_target: float,
    ctx: _MCMCContext,
    log_target: _LogTarget,
    rng: np.random.Generator,
) -> tuple[tuple[_MarsTerm, ...], float]:
    proposal, log_q_forward, log_q_reverse = _propose_structure_state(state, ctx, rng)
    proposal_log_target = log_target(proposal)
    log_alpha = proposal_log_target - state_log_target + log_q_reverse - log_q_forward
    if np.log(max(float(rng.random()), 1e-300)) < min(0.0, float(log_alpha)):
        return proposal, proposal_log_target
    return state, state_log_target


def _propose_structure_state(
    state: tuple[_MarsTerm, ...],
    ctx: _MCMCContext,
    rng: np.random.Generator,
) -> tuple[tuple[_MarsTerm, ...], float, float]:
    move = _sample_move(len(state), ctx.max_terms, rng)
    if move == "birth":
        return _propose_birth(state, ctx, rng)
    if move == "death":
        return _propose_death(state, ctx, rng)
    return _propose_change(state, ctx, rng)


def _propose_birth(
    state: tuple[_MarsTerm, ...],
    ctx: _MCMCContext,
    rng: np.random.Generator,
) -> tuple[tuple[_MarsTerm, ...], float, float]:
    term = ctx.proposal.sample_excluding(set(state), rng)
    if term is None:
        return state, 0.0, 0.0
    new_state = _state_key((*state, term))
    log_q_forward = math.log(_move_prob("birth", len(state), ctx.max_terms)) + ctx.proposal.log_prob_excluding(term, set(state))
    log_q_reverse = math.log(_move_prob("death", len(new_state), ctx.max_terms)) - math.log(len(new_state))
    return new_state, float(log_q_forward), float(log_q_reverse)


def _propose_death(
    state: tuple[_MarsTerm, ...],
    ctx: _MCMCContext,
    rng: np.random.Generator,
) -> tuple[tuple[_MarsTerm, ...], float, float]:
    idx = int(rng.integers(0, len(state)))
    term = state[idx]
    new_state = _drop_term(state, idx)
    log_q_forward = math.log(_move_prob("death", len(state), ctx.max_terms)) - math.log(len(state))
    log_q_reverse = math.log(_move_prob("birth", len(new_state), ctx.max_terms)) + ctx.proposal.log_prob_excluding(term, set(new_state))
    return new_state, float(log_q_forward), float(log_q_reverse)


def _propose_change(
    state: tuple[_MarsTerm, ...],
    ctx: _MCMCContext,
    rng: np.random.Generator,
) -> tuple[tuple[_MarsTerm, ...], float, float]:
    idx = int(rng.integers(0, len(state)))
    old_term = state[idx]
    kept = _drop_term(state, idx)
    new_term = ctx.proposal.sample_excluding(set(kept), rng)
    if new_term is None:
        return state, 0.0, 0.0
    new_state = _state_key((*kept, new_term))
    log_q_forward = _change_log_q(state, kept, new_term, ctx)
    log_q_reverse = _change_log_q(new_state, kept, old_term, ctx)
    return new_state, float(log_q_forward), float(log_q_reverse)


def _change_log_q(
    state: tuple[_MarsTerm, ...],
    kept: tuple[_MarsTerm, ...],
    term: _MarsTerm,
    ctx: _MCMCContext,
) -> float:
    return float(math.log(_move_prob("change", len(state), ctx.max_terms)) - math.log(len(state)) + ctx.proposal.log_prob_excluding(term, set(kept)))


def _sample_move(k: int, max_terms: int, rng: np.random.Generator) -> str:
    probs = [_move_prob(name, k, max_terms) for name in ("birth", "death", "change")]
    return str(rng.choice(("birth", "death", "change"), p=np.asarray(probs, dtype=float)))


def _move_prob(name: str, k: int, max_terms: int) -> float:
    probs = _structure_move_probs(k, max_terms)
    return {"birth": probs[0], "death": probs[1], "change": probs[2]}[name]


def _structure_move_probs(k: int, max_terms: int) -> tuple[float, float, float]:
    if k <= 0:
        return 1.0, 0.0, 0.0
    if k >= max_terms:
        return 0.0, 0.5, 0.5
    return 0.4, 0.3, 0.3


def _keep_mcmc_step(step: int, cfg: BayesianMarsSurrogateConfig) -> bool:
    if step < int(cfg.mcmc_burn_in):
        return False
    return (step - int(cfg.mcmc_burn_in)) % int(cfg.mcmc_thin) == 0


def _selected_states(
    samples: list[tuple[_MarsTerm, ...]],
    best_state: tuple[_MarsTerm, ...],
    limit: int,
) -> list[tuple[_MarsTerm, ...]]:
    counts = Counter(samples)
    selected = [state for state, _ in counts.most_common(int(limit))]
    if best_state not in selected:
        selected[-1] = best_state
    return selected


def _basis_state_weights(
    samples: list[tuple[_MarsTerm, ...]],
    selected: list[tuple[_MarsTerm, ...]],
) -> np.ndarray:
    counts = Counter(samples)
    raw = np.asarray([counts.get(state, 0) for state in selected], dtype=float)
    raw[raw <= 0.0] = 1.0
    return raw / float(np.sum(raw))


def _fit_selected_models(
    x: np.ndarray,
    y: np.ndarray,
    states: list[tuple[_MarsTerm, ...]],
    cfg: BayesianMarsSurrogateConfig,
    y_var: np.ndarray | None,
) -> tuple[_FittedBayesianMarsModel, ...]:
    return tuple(_fit_bayesian_mars(x, y, cfg, terms=state, y_var=y_var) for state in states)


def _design_for_state(x: np.ndarray, state: tuple[_MarsTerm, ...]) -> np.ndarray:
    if not state:
        return np.ones((x.shape[0], 1), dtype=float)
    return np.column_stack([np.ones((x.shape[0],), dtype=float), *(term.eval(x) for term in state)])


def _state_key(state: tuple[_MarsTerm, ...]) -> tuple[_MarsTerm, ...]:
    return tuple(sorted(state, key=_term_key))


def _term_key(term: _MarsTerm) -> tuple[tuple[int, float, int], ...]:
    return tuple((int(factor.feature), float(factor.knot), int(factor.side)) for factor in term.factors)


def _drop_term(state: tuple[_MarsTerm, ...], idx: int) -> tuple[_MarsTerm, ...]:
    return tuple(term for j, term in enumerate(state) if j != int(idx))


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


def _prior_inclusion(
    cfg: BayesianMarsSurrogateConfig,
    *,
    max_terms: int,
) -> float:
    if cfg.mcmc_term_prior is not None:
        return float(cfg.mcmc_term_prior)
    expected_terms = max(1.0, 0.5 * float(max_terms))
    return min(0.5, max(1e-4, expected_terms / float(max(max_terms * 8, 1))))


def _build_mcmc_basis_pool(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: BayesianMarsSurrogateConfig,
) -> tuple[tuple[_MarsTerm, ...], np.ndarray]:
    proposal = _TermProposal.from_inputs(np.asarray(x, dtype=float), np.asarray(y_std, dtype=float).reshape(-1), cfg)
    terms = _sample_term_pool(proposal, _pool_limit(cfg, x.shape[0]))
    if not terms:
        return (), np.zeros((x.shape[0], 0), dtype=float)
    return terms, np.column_stack([term.eval(x) for term in terms])


def _sample_term_pool(proposal: _TermProposal, limit: int) -> tuple[_MarsTerm, ...]:
    rng = np.random.default_rng(0)
    terms: set[_MarsTerm] = set()
    for _ in range(max(1000, int(limit) * 20)):
        term = proposal.sample_excluding(terms, rng)
        if term is None:
            break
        terms.add(term)
        if len(terms) >= int(limit):
            break
    return _state_key(tuple(terms))


def _pool_limit(cfg: BayesianMarsSurrogateConfig, num_obs: int) -> int:
    if cfg.mcmc_pool_size is not None:
        return int(cfg.mcmc_pool_size)
    return min(max(8, int(cfg.basis.max_terms) * 8), max(1, int(num_obs) - 1), 256)


def _mcmc_move_probs(k: int, *, max_terms: int, pool_size: int) -> tuple[float, float]:
    del pool_size
    if k <= 0:
        return 1.0, 0.0
    if k >= max_terms:
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
    if p_add > 0.0 and (p_drop <= 0.0 or rng.random() < p_add):
        return _propose_index_add(state, selected, p_add, max_terms, pool_size, rng)
    return _propose_index_drop(state, p_drop, max_terms, pool_size, rng)


def _propose_index_add(
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


def _propose_index_drop(
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
