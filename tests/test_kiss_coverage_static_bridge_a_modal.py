"""Kiss coverage bridge tests for modal batch CLI paths."""

from __future__ import annotations

from types import SimpleNamespace


def test_kiss_bridge_modal_batches_batches(monkeypatch):
    import experiments.modal_batches_impl as mb

    monkeypatch.setattr(mb, "batches_submitter", lambda *a, **k: None)

    Fn = type("Fn", (), {"spawn": lambda self: None})
    Func = type("Func", (), {"lookup": staticmethod(lambda *_a, **_k: Fn())})
    monkeypatch.setattr(mb, "modal", SimpleNamespace(Function=Func))
    mb.batches("tag", "submit-missing", None)


def test_kiss_bridge_modal_batches_batches_all_branches(monkeypatch, capsys):
    import experiments.modal_batches_impl as mb

    FakeDict = type("FakeDict", (dict,), {"len": lambda self: len(self)})
    monkeypatch.setattr(mb, "_results_dict", lambda _tag: FakeDict())
    monkeypatch.setattr(mb, "_submitted_dict", lambda _tag: FakeDict())
    monkeypatch.setattr(mb, "batches_submitter", lambda *a, **k: None)
    monkeypatch.setattr(mb, "_collect", lambda _tag: None)
    monkeypatch.setattr(mb.modal.Dict, "delete", lambda name: None)

    spawn_count = {"count": 0}

    Fn = type(
        "Fn",
        (),
        {"spawn": lambda self: spawn_count.__setitem__("count", spawn_count["count"] + 1) or None},
    )
    Func = type("Func", (), {"lookup": staticmethod(lambda *_a, **_k: Fn())})
    monkeypatch.setattr(
        mb,
        "modal",
        SimpleNamespace(Function=Func, Dict=SimpleNamespace(delete=lambda name: None)),
    )

    mb.batches("tag", "submit-missing-force", None)

    mb.batches("tag", "status", None)
    captured = capsys.readouterr()
    assert "results_available" in captured.out

    mb.batches("tag", "collect", None)

    mb.batches("tag", "clean_up", None)

    mb.batches("tag", "work", None, 2)
    assert spawn_count["count"] == 2
