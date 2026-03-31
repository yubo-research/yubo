"""Direct coverage tests for experiments/modal_batches.py.

This file uses module-level imports to ensure kiss detects the coverage.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import experiments.modal_batches as mb
from experiments.modal_batches import batches, clean_up, status


class _FakeDict(dict):
    def len(self):
        return len(self)


def test_status_coverage(monkeypatch, capsys):
    monkeypatch.setattr(mb, "_results_dict", lambda: _FakeDict({"a": 1, "b": 2}))
    monkeypatch.setattr(mb, "_submitted_dict", lambda: _FakeDict({"x": 1}))

    status()

    captured = capsys.readouterr()
    assert "results_available = 2" in captured.out
    assert "submitted = 1" in captured.out


def test_clean_up_success_coverage(monkeypatch, capsys):
    deleted = []
    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: deleted.append(name))

    clean_up()

    captured = capsys.readouterr()
    assert "batches_dict" in deleted
    assert "submitted_dict" in deleted
    assert "CLEANUP: deleted dict name=batches_dict" in captured.out
    assert "CLEANUP: deleted dict name=submitted_dict" in captured.out


def test_clean_up_exception_coverage(monkeypatch, capsys):
    def _raise(name):
        raise RuntimeError(f"Failed: {name}")

    monkeypatch.setattr(mb.modal.Dict, "delete", _raise)

    clean_up()

    captured = capsys.readouterr()
    assert "CLEANUP: dict delete failed" in captured.out
    assert "batches_dict" in captured.out


def test_batches_status_branch(monkeypatch, capsys):
    monkeypatch.setattr(mb, "_results_dict", lambda: _FakeDict())
    monkeypatch.setattr(mb, "_submitted_dict", lambda: _FakeDict())

    batches("status", None, None)

    captured = capsys.readouterr()
    assert "results_available" in captured.out


def test_batches_collect_branch(monkeypatch):
    collected = []
    monkeypatch.setattr(mb, "collect", lambda: collected.append(True))

    batches("collect", None, None)

    assert collected == [True]


def test_batches_clean_up_branch(monkeypatch):
    cleaned = []
    monkeypatch.setattr(mb, "clean_up", lambda: cleaned.append(True))

    batches("clean_up", None, None)

    assert cleaned == [True]


def test_batches_submit_missing_branch(monkeypatch):
    submitted = []
    monkeypatch.setattr(mb, "batches_submitter", lambda tag, force=False: submitted.append((tag, force)))

    batches("submit-missing", "test_tag", None)

    assert submitted == [("test_tag", False)]


def test_batches_submit_missing_force_branch(monkeypatch):
    submitted = []
    monkeypatch.setattr(mb, "batches_submitter", lambda tag, force=False: submitted.append((tag, force)))

    batches("submit-missing-force", "test_tag", None)

    assert submitted == [("test_tag", True)]


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

    batches("work", None, 3)

    assert len(spawned) == 3
    captured = capsys.readouterr()
    assert "WORK: 0" in captured.out
    assert "WORK: 1" in captured.out
    assert "WORK: 2" in captured.out
