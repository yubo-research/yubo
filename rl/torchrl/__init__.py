from __future__ import annotations

import importlib
from typing import Any


def load_patches() -> Any:
    return importlib.import_module("rl.torchrl.patches")


def __getattr__(name: str) -> Any:
    if name == "patches":
        return load_patches()
    raise AttributeError(name)


__all__ = ["load_patches", "patches"]
