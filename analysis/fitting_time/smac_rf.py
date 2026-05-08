"""SMAC3 random forest surrogate (Hutter et al.; packaged ``smac`` + ``pyrfr``).

Wraps :class:`smac.model.random_forest.RandomForest` for plain numeric design matrices
``(N, d)``: builds a :class:`ConfigSpace.ConfigurationSpace` with one
:class:`ConfigSpace.UniformFloatHyperparameter` per column and fits the same model SMAC uses
for sequential model-based optimization (including inactive-value imputation hooks for
conditional spaces—unused when all parameters are independent floats).

Requires the ``smac``, ``ConfigSpace``, and ``pyrfr`` packages (``pip install smac``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.model.random_forest import RandomForest

__all__ = ["SMACRFConfig", "SMACRFSurrogate"]


def _column_bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = x.min(axis=0).astype(np.float64)
    hi = x.max(axis=0).astype(np.float64)
    span = np.maximum(hi - lo, 1e-12)
    pad = 1e-6 * span
    return lo - pad, hi + pad


def _build_config_space(lower: np.ndarray, upper: np.ndarray) -> ConfigurationSpace:
    space = ConfigurationSpace()
    for j in range(lower.shape[0]):
        space.add(UniformFloatHyperparameter(f"x{j}", float(lower[j]), float(upper[j])))
    return space


@dataclass(frozen=True)
class SMACRFConfig:
    """Hyperparameters passed through to :class:`smac.model.random_forest.RandomForest`."""

    n_trees: int = 10
    n_points_per_tree: int = -1
    ratio_features: float = 5.0 / 6.0
    min_samples_split: int = 3
    min_samples_leaf: int = 3
    max_depth: int = 2**20
    eps_purity: float = 1e-8
    max_nodes: int = 2**20
    bootstrapping: bool = True
    log_y: bool = False
    seed: int = 0
    bounds: Literal["auto"] | tuple[np.ndarray, np.ndarray] = "auto"
    """``"auto"``: per-column ``[min, max]`` from training ``x`` (with padding). Or ``(low, high)``
    arrays of shape ``(d,)`` matching the physical scale of ``x``."""

    pca_components: int | None = None
    """Forwarded to SMAC ``RandomForest``; use ``None`` to disable PCA (typical without instance features)."""


class SMACRFSurrogate:
    """SMAC3 ``RandomForest`` mean and predictive variance (diagonal / inter-tree)."""

    def __init__(self, config: SMACRFConfig | None = None) -> None:
        self.cfg = config or SMACRFConfig()
        self._model: RandomForest | None = None
        self._dim: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (N, d), got {x.shape}")
        if self.cfg.ratio_features > 1.0:
            raise ValueError("ratio_features must be <= 1.0 for SMAC3/pyrfr (values > 1 set max_features to 0).")

        d = x.shape[1]
        self._dim = d
        if self.cfg.bounds == "auto":
            lower, upper = _column_bounds(x)
        else:
            lower, upper = self.cfg.bounds
            lower = np.asarray(lower, dtype=np.float64).reshape(-1)
            upper = np.asarray(upper, dtype=np.float64).reshape(-1)
            if lower.shape[0] != d or upper.shape[0] != d:
                raise ValueError(f"bounds must have length {d}, got {lower.shape[0]}")

        cs = _build_config_space(lower, upper)
        self._model = RandomForest(
            configspace=cs,
            n_trees=self.cfg.n_trees,
            n_points_per_tree=self.cfg.n_points_per_tree,
            ratio_features=self.cfg.ratio_features,
            min_samples_split=self.cfg.min_samples_split,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_depth=self.cfg.max_depth,
            eps_purity=self.cfg.eps_purity,
            max_nodes=self.cfg.max_nodes,
            bootstrapping=self.cfg.bootstrapping,
            log_y=self.cfg.log_y,
            instance_features=None,
            pca_components=self.cfg.pca_components,
            seed=self.cfg.seed,
        )
        self._model.train(x, y)
        return {"meta": self._model.meta}

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None or self._dim is None:
            raise RuntimeError("Call fit() before predict().")
        x_test = np.asarray(x_test, dtype=np.float64)
        if x_test.ndim != 2:
            raise ValueError(f"Expected x_test with shape (N, d), got {x_test.shape}")
        if x_test.shape[1] != self._dim:
            raise ValueError(f"Expected d={self._dim}, got {x_test.shape[1]}")

        mean, var = self._model.predict(x_test)
        assert var is not None
        return mean.reshape(-1), var.reshape(-1)
