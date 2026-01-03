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
