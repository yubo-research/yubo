from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ConsoleEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    channel: str = "train"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@runtime_checkable
class ConsoleObserver(Protocol):
    async def on_step(self, turn_idx: int, step_data: dict[str, Any]) -> None: ...

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None: ...

    async def on_reward(self, reward: float, metrics: dict[str, Any]) -> None: ...

    async def on_event(self, event: ConsoleEvent) -> None: ...


_ACTIVE_OBSERVER: contextvars.ContextVar[ConsoleObserver | None] = contextvars.ContextVar(
    "yubo_active_console_observer",
    default=None,
)


def active_console_observer() -> ConsoleObserver | None:
    return _ACTIVE_OBSERVER.get()


@contextmanager
def use_console_observer(observer: ConsoleObserver):
    token = _ACTIVE_OBSERVER.set(observer)
    try:
        yield observer
    finally:
        _ACTIVE_OBSERVER.reset(token)


__all__ = [
    "ConsoleEvent",
    "ConsoleObserver",
    "_ACTIVE_OBSERVER",
    "active_console_observer",
    "use_console_observer",
]
