from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from llm.model_types import Completion


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class SampleBatch:
    request_id: str
    samples: list[Completion]
    model: str = "runtime.model"
    created: int = field(default_factory=lambda: int(time.time()))
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any | None = None

    @property
    def completions(self) -> list[Completion]:
        return self.samples


__all__ = ["SampleBatch", "TokenUsage"]
