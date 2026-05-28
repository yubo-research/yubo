from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from enn.enn.enn_class import EpistemicNearestNeighbors
from torch import nn

from sampling.gather_proj_t import GatherProjSpec

from .uhd_enn_config import ENNImputerConfig
from .uhd_enn_fit_helpers import fit_enn_params
from .uhd_enn_imputer_predict import ENNMinusImputerPredictMixin
from .uhd_enn_imputer_tell import ENNMinusImputerTellMixin

__all__ = [
    "ENNImputerConfig",
    "ENNMinusImputer",
    "EpistemicNearestNeighbors",
    "fit_enn_params",
]


class ENNMinusImputer(ENNMinusImputerTellMixin, ENNMinusImputerPredictMixin):
    """Impute mu_minus in UHD's negative phase using ENN on SparseJL embeddings.

    This is a prototype to test "paired-minus imputation" (docs/uhd_enn_2.md #1):
    - always evaluate real x+ (positive phase),
    - predict mu(x-) for most negative phases once warm,
    - periodically refresh with real x- to recalibrate.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        cfg: ENNImputerConfig,
        noise_nz_fn: Callable[[int, float], tuple[np.ndarray, np.ndarray]],
    ):
        self._module = module
        self._cfg = cfg
        self._noise_nz_fn = noise_nz_fn

        # We do NOT embed the initial parameters. ENN operates on distances in embedding
        # space, and distances are translation-invariant. We maintain a consistent
        # coordinate system by starting at z_base=0 and updating via delta embeddings.
        self._z_base = torch.zeros((self._cfg.d,), dtype=torch.float32, device=torch.device("cpu"))
        self._delta_z = None
        self._delta_x: np.ndarray | None = None

        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._num_new_since_fit = 0

        self._enn_model: object | None = None
        self._enn_params: object | None = None

        self._num_negative_phases = 0
        self._num_real_evals = 0
        self._num_imputed = 0

        # Standardization stats (computed at fit time)
        self._y_mean = 0.0
        self._y_std = 1.0

        # Calibration stats (real mu_minus vs predicted)
        self._abs_err_ema: float | None = None
        self._num_calib = 0

        # Cache last real mu_plus for the current antithetic pair (needed for delta target).
        self._last_mu_plus: float | None = None

        self._gather_spec: GatherProjSpec | None = None
        if self._cfg.embedder == "gather":
            dim_ambient = int(sum(p.numel() for p in self._module.parameters()))
            self._gather_spec = GatherProjSpec.make(
                dim_ambient=dim_ambient,
                d=int(self._cfg.d),
                t=int(self._cfg.gather_t),
                seed=int(self._cfg.jl_seed),
            )

    @property
    def num_candidates(self) -> int:
        """Number of candidate seeds for UCB selection (public; avoids callers using ``_cfg``)."""
        return int(self._cfg.num_candidates)

    @property
    def num_real_evals(self) -> int:
        return self._num_real_evals

    @property
    def num_imputed(self) -> int:
        return self._num_imputed

    @property
    def abs_err_ema(self) -> float | None:
        return self._abs_err_ema

    def calibrate_minus(self, *, mu_minus_real: float) -> None:
        """Update prediction error stats using a real mu_minus at current pair."""
        if self._enn_params is None:
            return
        if self._cfg.target == "mu_plus":
            return
        if self._cfg.target == "delta" and self._last_mu_plus is None:
            return
        try:
            err = _calibration_error(self, mu_minus_real)
        except (RuntimeError, ValueError, TypeError, ArithmeticError):
            return
        if self._abs_err_ema is None:
            self._abs_err_ema = float(err)
        else:
            b = float(self._cfg.err_ema_beta)
            self._abs_err_ema = b * float(self._abs_err_ema) + (1.0 - b) * float(err)
        self._num_calib += 1


def _calibration_error(imputer: ENNMinusImputer, mu_minus_real: float) -> float:
    if imputer._cfg.target == "delta":
        delta_hat, _delta_se = imputer._predict_y_current()
        delta_real = float(imputer._last_mu_plus) - float(mu_minus_real)
        return abs(float(delta_real) - float(delta_hat))
    mu_hat, _se_hat = imputer.predict_current()
    return abs(float(mu_minus_real) - float(mu_hat))
