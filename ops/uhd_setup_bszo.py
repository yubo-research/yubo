from __future__ import annotations


def run_bszo_loop(*args, **kwargs):
    from ops.uhd_setup_monolith_bszo import run_bszo_loop as _run_bszo_loop

    return _run_bszo_loop(*args, **kwargs)


def _run_bszo_iterations(*args, **kwargs):
    from common.im import im

    return im("ops.uhd_setup_bszo_core")._run_bszo_iterations(*args, **kwargs)
