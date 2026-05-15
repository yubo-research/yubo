from __future__ import annotations

from typing import Any

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.eggroll_runtime_core import as_bool as _runtime_as_bool


def eggroll_bool(value: Any, *, name: str) -> bool:
    return _runtime_as_bool(value, name=name, error_cls=NoSuchDesignerError, option_label="EggRoll option")


def unit_decay(value: Any, *, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise NoSuchDesignerError(f"EggRoll option '{name}' must be in the interval (0, 1].")
    return parsed
