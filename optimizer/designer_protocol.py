from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from common.telemetry import Telemetry
    from optimizer.datum import Datum


class Designer(Protocol):
    def __call__(
        self,
        data: list["Datum"],
        num_arms: int,
        *,
        telemetry: Optional["Telemetry"] = None,
    ) -> list: ...


def get_designer_algo_metrics(designer) -> dict[str, float]:
    """Extract optimizer-specific metrics from a designer. Returns {} if not supported."""
    if not hasattr(designer, "get_algo_metrics") or not callable(designer.get_algo_metrics):
        return {}
    try:
        out = designer.get_algo_metrics()
        return dict(out) if out else {}
    except Exception:
        return {}
