from __future__ import annotations

import subprocess
from types import SimpleNamespace

from click.testing import CliRunner

from ops.modal_batches import (
    clean_up,
    cli,
    collect,
    deploy,
    status,
    stop,
    submit,
    submit_force,
)


def test_kiss_tidy_b_ops_modal_batches_cli(monkeypatch):
    def _run(*a, **k):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr("ops.modal_batches.sys.exit", lambda *a, **k: None)
    monkeypatch.setattr("ops.modal_batches._run_modal", lambda *a, **k: None)
    r = CliRunner()
    for cmd, args in (
        (cli, ["deploy", "t"]),
        (deploy, ["deploy", "t"]),
        (submit, ["submit", "t", "prep_timing_sweep"]),
        (submit_force, ["submit-force", "t", "prep_timing_sweep"]),
        (collect, ["collect", "t"]),
        (status, ["status", "t"]),
        (stop, ["stop", "t"]),
        (clean_up, ["clean-up", "t"]),
    ):
        assert r.invoke(cmd, args).exit_code == 0
    deploy("t")
    submit("t", "prep_timing_sweep")
    submit_force("t", "prep_timing_sweep")
