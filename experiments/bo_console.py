"""Console output for BO runs. Re-exports from common.console."""

from __future__ import annotations

from common.console import (
    CMA_METRICS,
    MULTI_TURBO_METRICS,
    TURBO_METRICS,
    BOConsoleCollector,
    print_bo_footer,
    register_opt_metrics,
)

__all__ = [
    "BOConsoleCollector",
    "CMA_METRICS",
    "MULTI_TURBO_METRICS",
    "TURBO_METRICS",
    "print_bo_footer",
    "register_opt_metrics",
]
