from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from llm.episode_types import Case, Signal


@dataclass(frozen=True)
class RuntimeConfig:
    concurrency: int = 1
    timeout_s: float | None = None
    fail_reward: float = 0.0
    convert_exceptions: bool = False


class Episode(Protocol):
    name: str

    async def run(
        self,
        case: Case,
        policy: Any,
        sampling: dict[str, Any],
        runtime: RuntimeConfig,
    ) -> Signal: ...


__all__ = ["Episode", "RuntimeConfig"]
