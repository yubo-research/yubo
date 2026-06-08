from __future__ import annotations

from pathlib import Path
from typing import Protocol

from llm.model_types import AdapterRef, Completion, SampleCall
from llm.sample_batch import SampleBatch, TokenUsage


class Sampler(Protocol):
    async def sample(self, call: SampleCall) -> SampleBatch: ...


def adapter_path(spec: AdapterRef | None) -> Path | None:
    return None if spec is None else Path(spec.path)


__all__ = [
    "AdapterRef",
    "Completion",
    "SampleBatch",
    "SampleCall",
    "Sampler",
    "TokenUsage",
    "adapter_path",
]
