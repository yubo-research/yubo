from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from ops.enn_incremental_batches import (
    cli,
    collect,
    deploy,
    status,
    stop,
    submit,
    submit_force,
)


def test_kiss_tidy_b_ops_enn_incremental_batches_cli(monkeypatch):
    from modal_timing_sweep_test_support import FakeResultsDict

    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Dict.from_name",
        lambda *a, **k: FakeResultsDict(),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Function.from_name",
        lambda *a, **k: SimpleNamespace(spawn=lambda *sa, **sk: None, spawn_map=lambda *sa, **sk: None),
    )
    monkeypatch.setattr("ops.enn_incremental_batches._run_modal", lambda *a, **k: None)
    monkeypatch.setattr("ops.enn_incremental_batches.sys.exit", lambda *a, **k: None)
    r = CliRunner()
    for cmd, args in (
        (cli, ["deploy", "add_method", "t"]),
        (deploy, ["add_method", "t"]),
        (submit, ["add_method", "t", "--d", "2", "--num-reps", "1"]),
        (submit_force, ["add_method", "t", "--d", "2", "--num-reps", "1"]),
        (collect, ["add_method", "t"]),
        (status, ["add_method", "t"]),
        (stop, ["add_method", "t"]),
    ):
        assert r.invoke(cmd, args).exit_code == 0
