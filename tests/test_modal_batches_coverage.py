"""Direct coverage tests for experiments/modal_batches_impl.py.

This file uses module-level imports to ensure kiss detects the coverage.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import experiments.modal_batches_impl as mb
from experiments.modal_batches_impl import batches, clean_up, status


class _FakeDict(dict):
    def len(self):
        return len(self)


def test_status_coverage(monkeypatch, capsys):
    monkeypatch.setattr(mb, "_results_dict", lambda tag: _FakeDict({"a": 1, "b": 2}))
    monkeypatch.setattr(mb, "_submitted_dict", lambda tag: _FakeDict({"x": 1}))

    status("test")

    captured = capsys.readouterr()
    assert "results_available = 2" in captured.out
    assert "submitted = 1" in captured.out


def test_clean_up_success_coverage(monkeypatch, capsys):
    deleted = []
    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: deleted.append(name))

    clean_up("test")

    captured = capsys.readouterr()
    assert "batches_dict_test" in deleted
    assert "submitted_dict_test" in deleted
    assert "CLEANUP: deleted dict name=batches_dict_test" in captured.out
    assert "CLEANUP: deleted dict name=submitted_dict_test" in captured.out


def test_clean_up_exception_coverage(monkeypatch, capsys):
    def _raise(name):
        raise RuntimeError(f"Failed: {name}")

    monkeypatch.setattr(mb.modal.Dict, "delete", _raise)

    clean_up("test")

    captured = capsys.readouterr()
    assert "CLEANUP: dict delete failed" in captured.out
    assert "batches_dict_test" in captured.out


def test_batches_status_branch(monkeypatch, capsys):
    monkeypatch.setattr(mb, "_results_dict", lambda tag: _FakeDict())
    monkeypatch.setattr(mb, "_submitted_dict", lambda tag: _FakeDict())

    batches("test", "status", None, None)

    captured = capsys.readouterr()
    assert "results_available" in captured.out


def test_batches_collect_branch(monkeypatch):
    collected = []
    monkeypatch.setattr(mb, "_collect", lambda tag: collected.append(tag))

    batches("test", "collect", None, None)

    assert collected == ["test"]


def test_batches_clean_up_branch(monkeypatch):
    cleaned = []
    monkeypatch.setattr(mb, "clean_up", lambda tag: cleaned.append(tag))

    batches("test", "clean_up", None, None)

    assert cleaned == ["test"]


def test_batches_submit_missing_branch(monkeypatch):
    submitted = []
    monkeypatch.setattr(mb, "batches_submitter", lambda tag, batch_tag, force=False: submitted.append((tag, batch_tag, force)))

    batches("test", "submit-missing", "test_tag", None)

    assert submitted == [("test", "test_tag", False)]


def test_batches_submit_missing_force_branch(monkeypatch):
    submitted = []
    monkeypatch.setattr(mb, "batches_submitter", lambda tag, batch_tag, force=False: submitted.append((tag, batch_tag, force)))

    batches("test", "submit-missing-force", "test_tag", None)

    assert submitted == [("test", "test_tag", True)]


def test_batches_work_branch(monkeypatch, capsys):
    spawned = []

    class _Fn:
        def spawn(self):
            spawned.append(True)

    class _Func:
        @staticmethod
        def lookup(*_a, **_k):
            return _Fn()

    monkeypatch.setattr(mb, "modal", SimpleNamespace(Function=_Func, Dict=MagicMock()))

    batches("test", "work", None, 3)

    assert len(spawned) == 3
    captured = capsys.readouterr()
    assert "WORK: 0" in captured.out
    assert "WORK: 1" in captured.out
    assert "WORK: 2" in captured.out


def test_batches_stop_branch(monkeypatch, capsys):
    """Test the stop branch of the batches function."""
    stopped = []
    monkeypatch.setattr(mb, "stop", lambda tag: stopped.append(tag))

    batches("test", "stop", None, None)

    assert stopped == ["test"]


def test_stop_notfounderror_is_handled_per_function(monkeypatch, capsys):
    """Test that NotFoundError is handled per-function by stop().

    When a function raises NotFoundError, stop() should print a per-function
    "not found" message and continue trying remaining functions.
    """
    import modal.exception

    class _NotFoundError(Exception):
        pass

    calls_to_from_name = []

    def _raise_not_found(app_name, func_name):
        calls_to_from_name.append((app_name, func_name))
        raise _NotFoundError(f"Function not found: {func_name}")

    monkeypatch.setattr(modal.exception, "NotFoundError", _NotFoundError)
    monkeypatch.setattr(mb.modal.Function, "from_name", _raise_not_found)
    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: None)

    mb.stop("missing_app")

    captured = capsys.readouterr()
    # Each function should get a "not found" message
    assert "modal_batches_worker: not found" in captured.out
    assert "modal_batches_resubmitter: not found" in captured.out
    assert "modal_batch_deleter: not found" in captured.out
    # All functions should be attempted
    assert len(calls_to_from_name) == 3
