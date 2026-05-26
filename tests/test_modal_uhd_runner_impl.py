from __future__ import annotations


def test_run() -> None:
    from ops.modal_uhd_runner_impl import _early_reject_fields, _run_fields, run

    assert run is not None
    assert _run_fields is not None
    assert _early_reject_fields is not None
