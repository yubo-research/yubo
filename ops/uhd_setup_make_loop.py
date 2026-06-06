from __future__ import annotations


def make_loop(*args, **kwargs):
    from ops.uhd_setup_monolith_make_loop import make_loop as _make_loop

    return _make_loop(*args, **kwargs)
