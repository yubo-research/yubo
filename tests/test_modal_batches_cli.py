"""Coverage for ops/modal_batches.py (modal CLI wrapper).

Module-level import links this test module to ops.modal_batches for kiss coverage.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import ops.modal_batches as mb_mod


class _OkResult:
    returncode = 0


@pytest.fixture
def captured_modal(monkeypatch):
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, env=None, **_kwargs):
        e = {k: v for k, v in (env or {}).items() if isinstance(v, str)}
        calls.append((list(cmd), e))
        return _OkResult()

    monkeypatch.setattr(mb_mod.subprocess, "run", fake_run)
    return calls


def test_modal_batches_get_impl_path():
    assert mb_mod._get_impl_path() == "experiments/modal_batches_impl.py"


@pytest.mark.parametrize(
    ("invoke_args", "expected_modal_args"),
    [
        (["deploy", "t1"], ["modal", "deploy", "experiments/modal_batches_impl.py"]),
        (
            ["submit", "t1", "b1"],
            [
                "modal",
                "run",
                "experiments/modal_batches_impl.py::batches",
                "t1",
                "submit-missing",
                "b1",
            ],
        ),
        (
            ["submit-force", "t1", "b1"],
            [
                "modal",
                "run",
                "experiments/modal_batches_impl.py::batches",
                "t1",
                "submit-missing-force",
                "b1",
            ],
        ),
        (
            ["collect", "t1"],
            ["modal", "run", "experiments/modal_batches_impl.py::batches", "t1", "collect"],
        ),
        (
            ["status", "t1"],
            ["modal", "run", "experiments/modal_batches_impl.py::batches", "t1", "status"],
        ),
        (
            ["stop", "t1"],
            ["modal", "run", "experiments/modal_batches_impl.py::batches", "t1", "stop"],
        ),
        (
            ["clean-up", "t1"],
            ["modal", "run", "experiments/modal_batches_impl.py::batches", "t1", "clean_up"],
        ),
    ],
)
def test_modal_batches_commands_set_modal_tag_and_args(captured_modal, invoke_args, expected_modal_args):
    runner = CliRunner()
    res = runner.invoke(mb_mod.cli, invoke_args)
    assert res.exit_code == 0, res.output
    assert len(captured_modal) == 1
    cmd, env = captured_modal[0]
    assert cmd == expected_modal_args
    assert env.get("MODAL_TAG") == "t1"


def test_modal_batches_propagates_subprocess_exit_code(captured_modal, monkeypatch):
    class _Bad:
        returncode = 7

    monkeypatch.setattr(mb_mod.subprocess, "run", lambda *a, **k: _Bad())

    runner = CliRunner()
    res = runner.invoke(mb_mod.cli, ["status", "x"])
    assert res.exit_code == 7
