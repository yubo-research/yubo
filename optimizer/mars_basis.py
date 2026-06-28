from __future__ import annotations

from typing import NamedTuple

import numpy as np


class _HingeFactor(NamedTuple):
    feature: int
    knot: float
    side: int

    def eval(self, x: np.ndarray) -> np.ndarray:
        col = np.asarray(x[:, int(self.feature)], dtype=float)
        if int(self.side) > 0:
            return np.maximum(col - float(self.knot), 0.0)
        return np.maximum(float(self.knot) - col, 0.0)

    def eval_active(self, x: np.ndarray, index: dict[int, int]) -> np.ndarray:
        col = np.asarray(x[:, index[int(self.feature)]], dtype=float)
        if int(self.side) > 0:
            return np.maximum(col - float(self.knot), 0.0)
        return np.maximum(float(self.knot) - col, 0.0)


class _MarsTerm(NamedTuple):
    factors: tuple[_HingeFactor, ...]

    @property
    def features(self) -> tuple[int, ...]:
        return tuple(int(f.feature) for f in self.factors)

    def eval(self, x: np.ndarray) -> np.ndarray:
        out = np.ones((x.shape[0],), dtype=float)
        for factor in self.factors:
            out *= factor.eval(x)
        return out

    def eval_active(self, x: np.ndarray, index: dict[int, int]) -> np.ndarray:
        out = np.ones((x.shape[0],), dtype=float)
        for factor in self.factors:
            out *= factor.eval_active(x, index)
        return out

    def gradient_active(self, x: np.ndarray, index: dict[int, int], num_active: int) -> np.ndarray:
        grad = np.zeros((int(x.shape[0]), int(num_active)), dtype=float)
        vals = [factor.eval_active(x, index) for factor in self.factors]
        for i, factor in enumerate(self.factors):
            deriv = np.full((int(x.shape[0]),), float(factor.side), dtype=float)
            deriv[vals[i] <= 0.0] = 0.0
            for j, val in enumerate(vals):
                if i != j:
                    deriv *= val
            grad[:, index[int(factor.feature)]] += deriv
        return grad


def _coerce_scalar_y(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError("MarsSurrogate requires scalar y observations")
        return arr[:, 0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"MarsSurrogate requires y shape (n,) or (n,1), got {arr.shape}")


def _ridge_solve(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = float(ridge) * np.eye(phi.shape[1], dtype=float)
    reg[0, 0] = 0.0
    lhs = phi.T @ phi + reg
    rhs = phi.T @ y
    return _safe_solve(lhs, rhs)


def _safe_solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _standardize_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(y, dtype=float).reshape(-1)
    mean = float(np.mean(arr)) if arr.size else 0.0
    scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (arr - mean) / scale, mean, scale


def _standardized_y_var(y_var: np.ndarray | None, y_scale: float) -> np.ndarray | None:
    if y_var is None:
        return None
    arr = np.asarray(y_var, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return None
    return arr / max(float(y_scale) ** 2, 1e-12)


def _score_columns(cols: np.ndarray, target: np.ndarray) -> np.ndarray:
    cols_arr = np.asarray(cols, dtype=float)
    target_arr = np.asarray(target, dtype=float).reshape(-1)
    centered_cols = cols_arr - np.mean(cols_arr, axis=0, keepdims=True)
    centered_target = target_arr - float(np.mean(target_arr))
    denom = np.linalg.norm(centered_cols, axis=0) * max(float(np.linalg.norm(centered_target)), 1e-12)
    return np.abs(centered_cols.T @ centered_target) / np.maximum(denom, 1e-12)


def _screen_features(x: np.ndarray, y: np.ndarray, max_features: int | None) -> np.ndarray:
    num_dim = int(x.shape[1])
    if max_features is None or num_dim <= int(max_features):
        return np.arange(num_dim, dtype=np.int64)
    y0 = y - float(np.mean(y))
    x0 = x - np.mean(x, axis=0, keepdims=True)
    denom = np.maximum(np.linalg.norm(x0, axis=0) * max(float(np.linalg.norm(y0)), 1e-12), 1e-12)
    score = np.abs(x0.T @ y0) / denom
    k = min(int(max_features), num_dim)
    idx = np.argpartition(-score, k - 1)[:k]
    return np.asarray(idx[np.argsort(-score[idx])], dtype=np.int64)


def _select_top(values: np.ndarray, budget: int) -> np.ndarray:
    count = int(min(max(budget, 0), values.size))
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.argpartition(-values, count - 1)[:count]
    return np.asarray(idx[np.argsort(-values[idx])], dtype=np.int64)


def _build_main_basis(
    x: np.ndarray,
    features: np.ndarray,
    quantiles: np.ndarray,
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    features = np.asarray(features, dtype=np.int64)
    if features.size == 0:
        return [], []
    return _basis_from_terms(x, *_candidate_hinges(x, features, quantiles))


def _candidate_hinges(
    x: np.ndarray,
    features: np.ndarray,
    quantiles: np.ndarray,
) -> tuple[list[int], list[float], list[int]]:
    knots_by_feature = np.asarray(np.quantile(x[:, features], quantiles, axis=0), dtype=float)
    if knots_by_feature.ndim == 1:
        knots_by_feature = knots_by_feature.reshape(-1, features.size)
    term_features: list[int] = []
    term_knots: list[float] = []
    term_sides: list[int] = []
    for feature_col, feature in enumerate(features):
        for knot in np.unique(knots_by_feature[:, int(feature_col)]):
            if np.isfinite(knot):
                term_features.extend((int(feature), int(feature)))
                term_knots.extend((float(knot), float(knot)))
                term_sides.extend((1, -1))
    return term_features, term_knots, term_sides


def _basis_from_terms(
    x: np.ndarray,
    term_features: list[int],
    term_knots: list[float],
    term_sides: list[int],
) -> tuple[list[_MarsTerm], list[np.ndarray]]:
    if not term_features:
        return [], []
    feature_idx = np.asarray(term_features, dtype=np.int64)
    cols_all = np.maximum(
        (x[:, feature_idx] - np.asarray(term_knots, dtype=float).reshape(1, -1)) * np.asarray(term_sides, dtype=float).reshape(1, -1),
        0.0,
    )
    keep = np.linalg.norm(cols_all - np.mean(cols_all, axis=0, keepdims=True), axis=0) > 1e-12
    terms: list[_MarsTerm] = []
    cols: list[np.ndarray] = []
    for idx in np.flatnonzero(keep):
        factor = _HingeFactor(int(term_features[int(idx)]), float(term_knots[int(idx)]), int(term_sides[int(idx)]))
        terms.append(_MarsTerm((factor,)))
        cols.append(np.asarray(cols_all[:, int(idx)], dtype=float).reshape(int(x.shape[0])))
    return terms, cols
