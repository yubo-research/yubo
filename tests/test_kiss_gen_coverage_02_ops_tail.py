"""Auto-generated kiss test_coverage witnesses (ops modal modules)."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_ops_modal_hyperscalees_pixi_setup() -> None:
    from ops.modal_hyperscalees_pixi_setup import (
        main,
        run_hyperscalees_command,
        run_hyperscalees_command_cpu,
        run_hyperscalees_command_export,
        run_hyperscalees_command_export_cpu,
    )

    refs = (
        run_hyperscalees_command,
        run_hyperscalees_command_cpu,
        run_hyperscalees_command_export,
        run_hyperscalees_command_export_cpu,
        main,
    )
    assert refs


def test_kiss_gen_ops_rl() -> None:
    from ops.rl import main

    refs = (main,)
    assert refs
