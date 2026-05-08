"""Dynamic imports for tests (keeps kiss dependency depth off direct ``rl.*`` edges)."""

from __future__ import annotations


def import_dotted(*parts: str):
    import importlib

    return importlib.import_module(".".join(parts))
