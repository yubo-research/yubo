"""Coverage for ops/modal_batches.py (modal CLI wrapper).

Module-level import links this test module to ops.modal_batches for kiss coverage.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import ops.modal_batches as mb_mod
from tests.modal_cli_test_helpers import (
    assert_modal_command_calls,
    capture_subprocess_run,
)


@pytest.fixture
def captured_modal(monkeypatch):
    return capture_subprocess_run(mb_mod, monkeypatch)


def test_modal_batches_get_impl_path():
    assert mb_mod._get_impl_path() == "experiments/modal_batches_impl.py"


@pytest.mark.parametrize(
    ("invoke_args", "expected_modal_cmds"),
    [
        (["deploy", "t1"], [["modal", "deploy", "experiments/modal_batches_impl.py"]]),
        (
            ["submit", "t1", "b1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_batches_impl.py::batches",
                    "--tag",
                    "t1",
                    "--cmd",
                    "submit-missing",
                    "--batch-tag",
                    "b1",
                ],
            ],
        ),
        (
            ["submit-force", "t1", "b1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_batches_impl.py::batches",
                    "--tag",
                    "t1",
                    "--cmd",
                    "submit-missing-force",
                    "--batch-tag",
                    "b1",
                ],
            ],
        ),
        (
            ["collect", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_batches_impl.py::batches",
                    "--tag",
                    "t1",
                    "--cmd",
                    "collect",
                ],
            ],
        ),
        (
            ["status", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_batches_impl.py::batches",
                    "--tag",
                    "t1",
                    "--cmd",
                    "status",
                ],
            ],
        ),
        (
            ["stop", "t1"],
            [
                ["modal", "app", "stop", "yubo_t1"],
                [
                    "modal",
                    "run",
                    "experiments/modal_batches_impl.py::batches",
                    "--tag",
                    "t1",
                    "--cmd",
                    "stop",
                ],
            ],
        ),
        (
            ["clean-up", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_batches_impl.py::batches",
                    "--tag",
                    "t1",
                    "--cmd",
                    "clean_up",
                ],
            ],
        ),
    ],
)
def test_modal_batches_commands_set_modal_tag_and_args(captured_modal, invoke_args, expected_modal_cmds):
    runner = CliRunner()
    res = runner.invoke(mb_mod.cli, invoke_args)
    assert res.exit_code == 0, res.output
    assert_modal_command_calls(captured_modal, expected_modal_cmds, modal_tag="t1")


def test_modal_batches_propagates_subprocess_exit_code(captured_modal, monkeypatch):
    class _Bad:
        returncode = 7

    monkeypatch.setattr(mb_mod.subprocess, "run", lambda *a, **k: _Bad())

    runner = CliRunner()
    res = runner.invoke(mb_mod.cli, ["status", "x"])
    assert res.exit_code == 7
