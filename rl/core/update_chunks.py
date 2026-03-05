from __future__ import annotations

from collections.abc import Callable


def run_chunked_updates(total_updates: int, chunk_size: int, do_update: Callable[[], None]) -> int:
    remaining = int(max(0, total_updates))
    if remaining == 0:
        return 0
    chunk = int(max(1, chunk_size))
    executed = 0
    while remaining > 0:
        current_chunk = min(chunk, remaining)
        for _ in range(current_chunk):
            do_update()
            executed += 1
        remaining -= int(current_chunk)
    return int(executed)
