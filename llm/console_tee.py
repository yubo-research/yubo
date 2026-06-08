from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any

from llm.line_tee import LineRoutingTee


@contextmanager
def tee_stdout_to_exp(observer: Any):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = LineRoutingTee(old_stdout, observer.route_line)
    sys.stderr = LineRoutingTee(old_stderr, observer.route_line)
    try:
        with observer.output_to(old_stdout):
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


__all__ = ["tee_stdout_to_exp"]
