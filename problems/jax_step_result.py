from __future__ import annotations

from typing import Any, NamedTuple


class JaxStepResult(NamedTuple):
    obs: Any
    state: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: Any
