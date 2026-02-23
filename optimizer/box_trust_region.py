from __future__ import annotations

from typing import Any

from enn.turbo.turbo_trust_region import TurboTrustRegion


class FixedLengthTurboTrustRegion(TurboTrustRegion):
    def _apply_fixed_length(self) -> None:
        fixed_length = getattr(self.config, "fixed_length", None)
        if fixed_length is not None:
            self.length = float(fixed_length)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._apply_fixed_length()

    def update(self, y_obs: Any, y_incumbent: Any) -> None:
        super().update(y_obs, y_incumbent)
        self._apply_fixed_length()

    def restart(self, rng: Any | None = None) -> None:
        super().restart(rng=rng)
        self._apply_fixed_length()
