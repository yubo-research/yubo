"""Types for turbo_m_ref module."""

from typing import NamedTuple

import numpy as np


class TrustRegion(NamedTuple):
    """Trust region bounds and center."""

    x_center: np.ndarray
    lb: np.ndarray
    ub: np.ndarray


class CandidatesResult(NamedTuple):
    """Result from candidate generation."""

    X_cand: np.ndarray
    y_cand: object
    hypers: dict


class StandardizedFX(NamedTuple):
    """Standardized function values with stats."""

    fX: np.ndarray
    mu: float
    sigma: float
