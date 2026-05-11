from __future__ import annotations


def make_loop(*args, **kwargs):
    from common.im import im

    return im("ops.uhd_setup_make_loop_impl").make_loop(*args, **kwargs)
