"""Turbo ENN designer (lazy facade)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optimizer.turbo_enn_designer_impl import TurboENNDesigner

__all__ = ["TurboENNDesigner"]


def __getattr__(name: str):
    if name == "TurboENNDesigner":
        impl = importlib.import_module("optimizer.turbo_enn_designer_impl")
        return impl.TurboENNDesigner
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted(__all__)
