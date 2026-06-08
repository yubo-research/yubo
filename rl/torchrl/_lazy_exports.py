from __future__ import annotations

import importlib
from typing import Any


def resolve_export(exports: dict[str, tuple[str, str | None]], name: str) -> Any:
    spec = exports.get(name)
    if spec is None:
        raise AttributeError(name)
    module_name, attr = spec
    mod = importlib.import_module(module_name)
    return mod if attr is None else getattr(mod, attr)


__all__ = ["resolve_export"]
