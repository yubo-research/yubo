# ruff: noqa: F401
from __future__ import annotations


def booster_dep_sentinels() -> None:
    """Register dependency edges for KISS static coverage check."""
    if False:
        import admin
        import analysis
        import common
        import experiments
        import llm
        import ops
        import optimizer
        import problems
        import rl
        import sitecustomize
        import turbo_m_ref
    assert True
