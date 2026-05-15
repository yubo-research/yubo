from __future__ import annotations

import asyncio
from typing import Any

from llm.episodes import Case, Episode, RuntimeConfig, Signal, failure_signal


class EpisodeRunner:
    def __init__(self, runtime: RuntimeConfig | None = None) -> None:
        self.runtime = runtime or RuntimeConfig()

    async def run_batch(
        self,
        episode: Episode,
        cases: list[Case],
        policy: Any,
        sampling: dict[str, Any],
    ) -> list[Signal]:
        if not cases:
            return []
        semaphore = asyncio.Semaphore(max(1, int(self.runtime.concurrency)))

        async def run_one(case: Case) -> Signal:
            async with semaphore:
                try:
                    awaitable = episode.run(case, policy, sampling, self.runtime)
                    if self.runtime.timeout_s is None:
                        return await awaitable
                    return await asyncio.wait_for(awaitable, timeout=float(self.runtime.timeout_s))
                except asyncio.TimeoutError as exc:
                    if not self.runtime.convert_exceptions:
                        raise
                    return failure_signal(case, status="timeout", error=exc, reward=self.runtime.fail_reward)
                except Exception as exc:
                    if not self.runtime.convert_exceptions:
                        raise
                    return failure_signal(case, status="error", error=exc, reward=self.runtime.fail_reward)

        return await asyncio.gather(*(run_one(case) for case in cases))


__all__ = ["EpisodeRunner"]
