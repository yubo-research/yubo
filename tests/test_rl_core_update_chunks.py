from __future__ import annotations

from rl.core.update_chunks import run_chunked_updates


def test_run_chunked_updates_executes_exact_count() -> None:
    calls: list[int] = []

    def _do_update() -> None:
        calls.append(1)

    executed = run_chunked_updates(total_updates=7, chunk_size=3, do_update=_do_update)

    assert executed == 7
    assert len(calls) == 7


def test_run_chunked_updates_handles_zero_updates() -> None:
    calls: list[int] = []

    executed = run_chunked_updates(total_updates=0, chunk_size=4, do_update=lambda: calls.append(1))

    assert executed == 0
    assert calls == []
