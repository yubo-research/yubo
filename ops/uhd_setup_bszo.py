from __future__ import annotations


def run_bszo_loop(*args, **kwargs):
    from common.im import im

    return im("ops.uhd_setup_bszo_core").run_bszo_loop(*args, **kwargs)


def _run_bszo_iterations(*args, **kwargs):
    from common.im import im

    return im("ops.uhd_setup_bszo_core")._run_bszo_iterations(*args, **kwargs)
