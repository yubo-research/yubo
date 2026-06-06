from __future__ import annotations


def run_simple_loop(*args, **kwargs):
    from ops.uhd_setup import run_simple_loop as _run_simple_loop

    return _run_simple_loop(*args, **kwargs)
