from __future__ import annotations

import re
from collections import deque
from typing import Any

from common.console_core import _dim, _green

# ---- BO (opt) metrics ----

TURBO_METRICS = [
    ("tr_length", 8, ".3f"),
    ("tr_obs", 6, ".0f"),
    ("fit_dt", 6, ".3f"),
    ("select_dt", 6, ".3f"),
]
MULTI_TURBO_METRICS = [
    ("tr_length", 8, ".3f"),
    ("fit_dt", 6, ".3f"),
    ("select_dt", 6, ".3f"),
    ("region_idx", 6, ".0f"),
    ("region_alloc", 8, ".0f"),
]
CMA_METRICS = [("sigma", 7, ".4f")]

_OPT_SCHEMAS: dict[str, list[tuple[str, int, str]]] = {
    "cma": CMA_METRICS,
    "turbo-enn-multi": MULTI_TURBO_METRICS,
}


def register_opt_metrics(opt_name: str, metrics: list[tuple[str, int, str]]) -> None:
    """Register opt-specific metric columns. metrics: [(key, width, fmt), ...]"""
    _OPT_SCHEMAS[opt_name] = metrics


def _parse_iter_line(line: str) -> dict[str, Any] | None:
    """Parse ITER line; return dict of fields or None if not an ITER line."""
    s = line.strip()
    if not s.startswith("ITER:"):
        return None
    out: dict[str, Any] = {}
    for m in re.finditer(r"(\w+)\s*=\s*(\S+)", s[5:]):
        k, v = m.group(1), m.group(2)
        v_clean = v.rstrip("s")
        try:
            out[k] = float(v_clean)
        except ValueError:
            out[k] = v_clean
    if "iter" not in out:
        return None
    return out


def print_bo_footer(best_return: float, total_time: float) -> None:
    """Print run completion summary."""
    print(flush=True)
    print(_dim("-" * 72), flush=True)
    print(_green(f"Done  ret_best={best_return:.1f}  time={total_time:.1f}s"), flush=True)
    print(_dim("-" * 72), flush=True)


class BOConsoleCollector:
    """Collector that echoes ITER lines to stdout (same format as data file). Implements Collector interface."""

    def __init__(self, *, inner: Any = None):
        self._lines = deque()
        self._inner = inner
        self._header_printed = False
        self._pre_header_buffer: list[str] = []
        self._best_return = -1e99

    def _flush_pre_header(self) -> None:
        """Print buffered lines (PROBLEM, algo output)."""
        for buf_line in self._pre_header_buffer:
            print(buf_line, flush=True)
        self._pre_header_buffer.clear()
        self._header_printed = True

    def __call__(self, line: str) -> None:
        self._lines.append(line)
        parsed = _parse_iter_line(line)
        if parsed is not None:
            self._flush_pre_header()
            try:
                r = float(parsed.get("ret_best", parsed.get("ret_eval", -1e99)))
                if r > self._best_return:
                    self._best_return = r
            except (TypeError, ValueError):
                pass
            print(line, flush=True)
        else:
            if self._header_printed:
                print(line, flush=True)
            else:
                self._pre_header_buffer.append(line)
        if self._inner is not None:
            self._inner(line)

    def __iter__(self):
        return iter(self._lines)
