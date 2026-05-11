"""Lazy import helper."""

from __future__ import annotations

import importlib


def im(name: str):
    """Import a module by dotted name."""
    return importlib.import_module(name)
