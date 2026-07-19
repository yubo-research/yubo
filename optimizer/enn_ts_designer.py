"""ENN Thompson-sampling designer (lazy facade)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizer.enn_ts_designer_impl import EnnTSDesigner

__all__ = ["EnnTSDesigner"]


def __getattr__(name: str):
    if name == "EnnTSDesigner":
        impl = importlib.import_module("optimizer.enn_ts_designer_impl")
        return impl.EnnTSDesigner
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted(__all__)
