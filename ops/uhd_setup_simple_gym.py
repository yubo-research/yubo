from __future__ import annotations


def run_simple_loop(*args, **kwargs):
    from common.im import im

    return im("ops.uhd_setup_simple_gym_impl").run_simple_loop(*args, **kwargs)
