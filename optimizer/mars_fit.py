from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars_basis import (
    _HingeFactor,
    _MarsTerm,
    _ridge_solve,
    _screen_features,
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


@dataclass(frozen=True)
class _MarsPairCandidate:
    terms: tuple[_MarsTerm, _MarsTerm]
    cols: tuple[np.ndarray, np.ndarray]
    rss: float


@dataclass(frozen=True)
class _MarsSubmodel:
    terms: tuple[_MarsTerm, ...]
    cols: tuple[np.ndarray, ...]
    coef: np.ndarray
    rss: float
    gcv: float


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
    terms, cols = _forward_mars_terms(x_arr, y_std, cfg)
    if not terms:
        return _empty_model(y_mean, y_scale, num_dim)
    submodel = _backward_prune_by_gcv(y_std, cfg, terms, cols)
    return _FittedMarsModel(submodel.terms, submodel.coef, y_mean, y_scale, num_dim)


def _empty_model(y_mean: float, y_scale: float, num_dim: int) -> _FittedMarsModel:
    return _FittedMarsModel((), np.array([0.0]), y_mean, y_scale, int(num_dim))


def _forward_mars_terms(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    terms: list[_MarsTerm] = []
    cols: list[np.ndarray] = []
    features = _screen_features(x, y_std, cfg.feature_screen)
    knots = _candidate_knots_by_feature(x, features, cfg)
    rss = _rss_for_cols(cols, y_std, cfg)
    max_non_intercept = min(int(cfg.max_terms) - 1, max(1, int(x.shape[0]) - 1))
    while len(terms) + 2 <= max_non_intercept:
        candidate = _best_forward_pair(x, y_std, cfg, terms, cols, knots)
        if candidate is None or rss - candidate.rss <= float(cfg.min_rss_improvement):
            break
        terms.extend(candidate.terms)
        cols.extend(candidate.cols)
        rss = candidate.rss
    return terms, cols


def _candidate_knots_by_feature(
    x: np.ndarray,
    features: np.ndarray,
    cfg: MarsSurrogateConfig,
) -> dict[int, np.ndarray]:
    return {int(feature): _candidate_knots(x[:, int(feature)], cfg) for feature in features}


def _candidate_knots(x_feature: np.ndarray, cfg: MarsSurrogateConfig) -> np.ndarray:
    values = np.unique(np.asarray(x_feature, dtype=float)[np.isfinite(x_feature)])
    if values.size <= 2:
        return np.zeros((0,), dtype=float)
    interior = values[1:-1]
    if cfg.knots_per_feature is None or interior.size <= int(cfg.knots_per_feature):
        return interior
    quantiles = np.linspace(0.0, 1.0, int(cfg.knots_per_feature) + 2)[1:-1]
    return np.unique(np.quantile(interior, quantiles))


def _best_forward_pair(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
    terms: list[_MarsTerm],
    cols: list[np.ndarray],
    knots: dict[int, np.ndarray],
) -> _MarsPairCandidate | None:
    best: _MarsPairCandidate | None = None
    existing = set(terms)
    parent_items = _parent_items(x, terms, cols, cfg)
    for parent, parent_col in parent_items:
        for feature, feature_knots in knots.items():
            if parent is not None and int(feature) in parent.features:
                continue
            for knot in feature_knots:
                candidate = _score_forward_pair(x, y_std, cfg, cols, existing, parent, parent_col, int(feature), float(knot))
                if candidate is not None and (best is None or candidate.rss < best.rss):
                    best = candidate
    return best


def _parent_items(
    x: np.ndarray,
    terms: list[_MarsTerm],
    cols: list[np.ndarray],
    cfg: MarsSurrogateConfig,
) -> list[tuple[_MarsTerm | None, np.ndarray]]:
    out: list[tuple[_MarsTerm | None, np.ndarray]] = [(None, np.ones((x.shape[0],), dtype=float))]
    if int(cfg.interaction_order) <= 1:
        return out
    out.extend((term, col) for term, col in zip(terms, cols, strict=True) if len(term.factors) < int(cfg.interaction_order))
    return out


def _score_forward_pair(
    x: np.ndarray,
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
    cols: list[np.ndarray],
    existing: set[_MarsTerm],
    parent: _MarsTerm | None,
    parent_col: np.ndarray,
    feature: int,
    knot: float,
) -> _MarsPairCandidate | None:
    terms_pair = _paired_terms(parent, feature, knot)
    if terms_pair[0] in existing or terms_pair[1] in existing:
        return None
    col_pair = tuple(parent_col * factor.eval(x) for factor in terms_pair)
    if not all(_usable_col(col) for col in col_pair):
        return None
    rss = _rss_for_cols([*cols, *col_pair], y_std, cfg)
    return _MarsPairCandidate(terms_pair, col_pair, rss)


def _paired_terms(parent: _MarsTerm | None, feature: int, knot: float) -> tuple[_MarsTerm, _MarsTerm]:
    prefix = () if parent is None else parent.factors
    right = _HingeFactor(int(feature), float(knot), 1)
    left = _HingeFactor(int(feature), float(knot), -1)
    return _MarsTerm(tuple((*prefix, right))), _MarsTerm(tuple((*prefix, left)))


def _usable_col(col: np.ndarray) -> bool:
    centered = np.asarray(col, dtype=float) - float(np.mean(col))
    return bool(np.linalg.norm(centered) > 1e-12)


def _backward_prune_by_gcv(
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
    terms: list[_MarsTerm],
    cols: list[np.ndarray],
) -> _MarsSubmodel:
    current_terms = tuple(terms)
    current_cols = tuple(cols)
    best = _submodel(current_terms, current_cols, y_std, cfg)
    while current_terms:
        current = _best_deletion(current_terms, current_cols, y_std, cfg)
        if current.gcv < best.gcv:
            best = current
        current_terms = current.terms
        current_cols = current.cols
    return best


def _best_deletion(
    terms: tuple[_MarsTerm, ...],
    cols: tuple[np.ndarray, ...],
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
) -> _MarsSubmodel:
    candidates = (_submodel(_drop_index(terms, idx), _drop_index(cols, idx), y_std, cfg) for idx in range(len(terms)))
    return min(candidates, key=lambda item: item.gcv)


def _drop_index(items: tuple, idx: int) -> tuple:
    return tuple(item for j, item in enumerate(items) if j != int(idx))


def _submodel(
    terms: tuple[_MarsTerm, ...],
    cols: tuple[np.ndarray, ...],
    y_std: np.ndarray,
    cfg: MarsSurrogateConfig,
) -> _MarsSubmodel:
    phi = _design_from_cols(cols, y_std.shape[0])
    coef = _ridge_solve(phi, y_std, float(cfg.ridge))
    residual = y_std - phi @ coef
    rss = float(residual @ residual)
    return _MarsSubmodel(terms, cols, coef, rss, _gcv(rss, y_std.shape[0], len(terms), cfg))


def _rss_for_cols(cols: list[np.ndarray], y_std: np.ndarray, cfg: MarsSurrogateConfig) -> float:
    phi = _design_from_cols(tuple(cols), y_std.shape[0])
    coef = _ridge_solve(phi, y_std, float(cfg.ridge))
    residual = y_std - phi @ coef
    return float(residual @ residual)


def _design_from_cols(cols: tuple[np.ndarray, ...], num_obs: int) -> np.ndarray:
    if not cols:
        return np.ones((int(num_obs), 1), dtype=float)
    return np.column_stack([np.ones((int(num_obs),), dtype=float), *cols])


def _gcv(rss: float, num_obs: int, num_terms: int, cfg: MarsSurrogateConfig) -> float:
    effective_params = 1.0 + float(num_terms) + float(cfg.gcv_penalty) * float(num_terms) / 2.0
    denom = 1.0 - effective_params / float(max(1, num_obs))
    if denom <= 1e-12:
        return float("inf")
    return float(rss) / (float(max(1, num_obs)) * denom**2)
