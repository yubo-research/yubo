"""Regression tests for ENN incremental batch client behavioral bugs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_client_clean_up_fails_closed_on_dict_delete_error(monkeypatch):
    import experiments.modal_enn_incremental_batches_client as client
    import experiments.modal_enn_incremental_batches_common as common

    def _run(cmd, **_kwargs):
        if len(cmd) >= 3 and cmd[1:3] == ["dict", "delete"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common.subprocess, "run", _run)

    with pytest.raises(SystemExit):
        client.clean_up("add_method-t1")


def test_impl_clean_up_fails_closed_on_dict_delete_error(monkeypatch):
    import experiments.modal_enn_incremental_batches_common as common
    import experiments.modal_enn_incremental_batches_impl as impl

    def _run(cmd, **_kwargs):
        if len(cmd) >= 3 and cmd[1:3] == ["dict", "delete"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common.subprocess, "run", _run)

    with pytest.raises(SystemExit):
        impl.clean_up("add_method-t1")


def test_client_stop_stops_modal_app_not_only_dicts(monkeypatch):
    import experiments.modal_enn_incremental_batches_client as client
    import experiments.modal_enn_incremental_batches_common as common

    calls: list[str] = []

    def _run(cmd, **_kwargs):
        if len(cmd) >= 2 and cmd[1] == "app":
            calls.append("app_stop")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common.subprocess, "run", _run)

    client.stop("add_method-t1")

    assert "app_stop" in calls


def test_stop_deletes_dicts_after_failed_app_stop(monkeypatch):
    import experiments.modal_enn_incremental_batches_common as common

    dict_deletes: list[str] = []

    def _run(cmd, **_kwargs):
        if len(cmd) >= 2 and cmd[1] == "app":
            return SimpleNamespace(returncode=1)
        if len(cmd) >= 3 and cmd[1:3] == ["dict", "delete"]:
            dict_deletes.append(cmd[5])
        return SimpleNamespace(returncode=0)

    with pytest.raises(SystemExit):
        common.stop("add_method-t1", run=_run)

    assert dict_deletes == [
        "enn_incremental_results_add_method-t1",
        "enn_incremental_submitted_add_method-t1",
    ]
