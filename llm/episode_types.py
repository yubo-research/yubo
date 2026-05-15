from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    prompt: Any
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    kind: str
    text: str
    name: str | None = None
    latency_s: float = 0.0
    tokens: int = 0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    reward: float
    status: str
    turns: tuple[Turn, ...] = ()
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


__all__ = ["Case", "Signal", "Turn"]
