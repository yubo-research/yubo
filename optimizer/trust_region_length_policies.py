"""
Length-update policies for shaped trust regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

UpdateMode = Literal["option_a", "option_b", "option_c"]


class _LengthPolicy:
    def reset(self) -> None:
        return

    def set_acceptance_ratio(self, *, pred: float, act: float, boundary_hit: bool) -> None:
        _ = pred, act, boundary_hit
        return

    def update_length(
        self,
        *,
        tr: Any,
        improved: bool | None,
        base_length: float,
        fixed_length: float | None,
        length_max: float,
    ) -> float:
        _ = base_length, fixed_length, length_max
        if improved is None:
            return float(tr.length)
        tr._update_counters_and_length(improved=improved)
        return float(tr.length)


@dataclass
class _OptionCLengthPolicy(_LengthPolicy):
    rho_bad: float
    rho_good: float
    gamma_down: float
    gamma_up: float
    _pending_rho: float | None = field(default=None, init=False)
    _pending_boundary_hit: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._pending_rho = None
        self._pending_boundary_hit = False

    def set_acceptance_ratio(self, *, pred: float, act: float, boundary_hit: bool) -> None:
        eps = 1e-12
        denom = pred
        if not np.isfinite(denom) or abs(float(denom)) < eps:
            denom = eps if float(pred) >= 0.0 else -eps
        self._pending_rho = float(act) / float(denom)
        self._pending_boundary_hit = bool(boundary_hit)

    def update_length(
        self,
        *,
        tr: Any,
        improved: bool | None,
        base_length: float,
        fixed_length: float | None,
        length_max: float,
    ) -> float:
        _ = tr, improved
        rho = self._pending_rho
        length = float(base_length if fixed_length is None else tr.length)
        if rho is None:
            return float(length)
        if rho < float(self.rho_bad):
            length *= float(self.gamma_down)
        elif rho > float(self.rho_good) and self._pending_boundary_hit:
            length *= float(self.gamma_up)
        self._pending_rho = None
        self._pending_boundary_hit = False
        return float(min(length, length_max))
