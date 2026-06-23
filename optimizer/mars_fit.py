from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars_basis import (
    _build_main_basis,
    _MarsTerm,
    _ridge_solve,
    _score_columns,
    _screen_features,
    _select_top,
    _standardize_y,
)
from .mars_config import MarsSurrogateConfig
from .mars_geometry import _low_rank_factor_from_isotropic_spectrum, _LowRankFactor


@dataclass
class _FittedMarsModel:
    terms: tuple[_MarsTerm, ...]
    coef: np.ndarray
    y_mean: float
    y_scale: float
    num_dim: int

    def _design(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        phi = np.ones((x_arr.shape[0], len(self.terms) + 1), dtype=float)
        for j, term in enumerate(self.terms, start=1):
            phi[:, j] = term.eval(x_arr)
        return phi

    def predict(self, x: np.ndarray) -> np.ndarray:
        return float(self.y_mean) + float(self.y_scale) * (self._design(x) @ self.coef)

    def active_features(self) -> tuple[int, ...]:
        out: set[int] = set()
        for term in self.terms:
            out.update(term.features)
        return tuple(sorted(out))

    def active_low_rank_factor(
        self,
        cfg: MarsSurrogateConfig,
        rng: np.random.Generator,
    ) -> _LowRankFactor | None:
        features = self.active_features()
        if not features:
            return None
        grad = _active_gradient(self, cfg, features, rng)
        if grad is None:
            return None
        return _low_rank_from_gradient(self, cfg, features, grad)


def _active_gradient(
    model: _FittedMarsModel,
    cfg: MarsSurrogateConfig,
    features: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray | None:
    x_active = rng.uniform(0.0, 1.0, size=(int(cfg.active_samples), len(features)))
    index = {feature: i for i, feature in enumerate(features)}
    grad = np.zeros_like(x_active, dtype=float)
    for coef, term in zip(model.coef[1:], model.terms, strict=False):
        if coef != 0.0:
            grad += float(coef) * term.gradient_active(x_active, index, len(features))
    grad *= float(model.y_scale)
    if not np.any(np.isfinite(grad)) or np.linalg.norm(grad) <= 0.0:
        return None
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _low_rank_from_gradient(
    model: _FittedMarsModel,
    cfg: MarsSurrogateConfig,
    features: tuple[int, ...],
    grad: np.ndarray,
) -> _LowRankFactor | None:
    rank = min(int(cfg.active_rank), len(features), int(cfg.active_samples))
    if rank <= 0:
        return None
    try:
        _, singular, vt = np.linalg.svd(grad, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    eigvals = (singular**2) / float(max(1, grad.shape[0]))
    keep = int(min(rank, eigvals.size))
    if keep <= 0:
        return None
    basis = np.zeros((int(model.num_dim), keep), dtype=float)
    for local_col, feature in enumerate(features):
        basis[int(feature), :] = vt[:keep, local_col]
    return _low_rank_factor_from_isotropic_spectrum(
        alpha_base=0.0,
        basis=basis,
        extra_eigvals=eigvals[:keep],
        dim=int(model.num_dim),
        lam_min=float(cfg.lam_min),
        lam_max=float(cfg.lam_max),
        eps=float(cfg.eps),
        rank_cap=keep,
        kappa_max=float(cfg.kappa_max),
    )


def _fit_single_mars(
    x: np.ndarray,
    y: np.ndarray,
    cfg: MarsSurrogateConfig,
) -> _FittedMarsModel:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    num_obs, num_dim = x_arr.shape
    y_std, y_mean, y_scale = _standardize_y(y_arr)
    if num_obs < 2 or int(cfg.max_terms) <= 1:
        return _empty_model(y_mean, y_scale, num_dim)
    main_terms, main_cols = _candidate_main_basis(x_arr, y_std, cfg)
    if not main_terms:
        return _empty_model(y_mean, y_scale, num_dim)
    terms, cols = _select_main_basis(main_terms, main_cols, y_std, cfg, num_obs)
    terms, cols = _add_interactions(terms, cols, y_std, cfg, num_obs)
    phi = np.column_stack([np.ones((num_obs,), dtype=float), *cols])
    coef = _ridge_solve(phi, y_std, float(cfg.ridge))
    return _FittedMarsModel(tuple(terms), coef, y_mean, y_scale, num_dim)


def _empty_model(y_mean: float, y_scale: float, num_dim: int) -> _FittedMarsModel:
    return _FittedMarsModel((), np.array([0.0]), y_mean, y_scale, int(num_dim))


def _candidate_main_basis(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    features = _screen_features(x, y_std, int(cfg.feature_screen))
    quantiles = np.linspace(0.2, 0.8, int(cfg.knots_per_feature))
    return _build_main_basis(x, features, quantiles)


def _term_budgets(cfg: MarsSurrogateConfig, num_obs: int) -> tuple[int, int]:
    max_non_intercept = min(int(cfg.max_terms) - 1, max(1, int(num_obs) - 1))
    if int(cfg.interaction_order) == 1:
        return max_non_intercept, 0
    main_budget = max(1, max_non_intercept // 2)
    return main_budget, max_non_intercept - main_budget


def _select_main_basis(
    main_terms: list[_MarsTerm],
    main_cols: list[np.ndarray],
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
    num_obs: int,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    main_budget, _ = _term_budgets(cfg, num_obs)
    main_mat = np.column_stack(main_cols)
    main_idx = _select_top(_score_columns(main_mat, y_std), main_budget)
    terms = [main_terms[int(i)] for i in main_idx]
    cols = [main_mat[:, int(i)] for i in main_idx]
    return terms, cols


def _add_interactions(
    terms: list[_MarsTerm],
    cols: list[np.ndarray],
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
    num_obs: int,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    _, inter_budget = _term_budgets(cfg, num_obs)
    if inter_budget <= 0 or len(cols) < 2:
        return terms, cols
    residual = _main_residual(cols, y_std, cfg, num_obs)
    inter_terms, inter_cols = _interaction_basis(terms, cols)
    if not inter_terms:
        return terms, cols
    inter_mat = np.column_stack(inter_cols)
    inter_idx = _select_top(_score_columns(inter_mat, residual), inter_budget)
    terms.extend(inter_terms[int(i)] for i in inter_idx)
    cols.extend(inter_mat[:, int(i)] for i in inter_idx)
    return terms, cols


def _main_residual(
    cols: list[np.ndarray],
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
    num_obs: int,
) -> np.ndarray:
    phi_main = np.column_stack([np.ones((num_obs,), dtype=float), *cols])
    coef_main = _ridge_solve(phi_main, y_std, float(cfg.ridge))
    return y_std - phi_main @ coef_main


def _interaction_basis(
    terms: list[_MarsTerm],
    cols: list[np.ndarray],
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    inter_terms: list[_MarsTerm] = []
    inter_cols: list[np.ndarray] = []
    for i in range(len(terms)):
        feat_i = set(terms[i].features)
        for j in range(i + 1, len(terms)):
            if not feat_i.intersection(terms[j].features):
                inter_terms.append(_MarsTerm(tuple(terms[i].factors + terms[j].factors)))
                inter_cols.append(cols[i] * cols[j])
    return inter_terms, inter_cols
