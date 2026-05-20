"""Regression: ops CLI stop and common.stop should use the same dict cleanup."""

from __future__ import annotations

from types import SimpleNamespace


def test_common_stop_deletes_dicts_via_subprocess_like_ops_cli(monkeypatch):
    import experiments.modal_enn_incremental_batches_common as common

    subprocess_cmds: list[list[str]] = []
    sdk_deletes: list[tuple] = []

    def _run(cmd, **_kwargs):
        subprocess_cmds.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common, "_stop_modal_app", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        common.modal.Dict.objects,
        "delete",
        lambda *args, **kwargs: sdk_deletes.append((args, kwargs)),
    )

    common.stop("add_method-t1", run=_run)

    dict_deletes = [c for c in subprocess_cmds if len(c) >= 3 and c[1:3] == ["dict", "delete"]]
    assert len(dict_deletes) == 2
    assert sdk_deletes == []
