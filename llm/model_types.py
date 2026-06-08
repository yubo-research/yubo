from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdapterRef:
    name: str
    adapter_id: int
    path: str

    @classmethod
    def from_tuple(cls, spec: tuple[str, int, str] | None) -> "AdapterRef | None":
        if spec is None:
            return None
        name, adapter_id, path = spec
        return cls(name=str(name), adapter_id=int(adapter_id), path=str(path))

    def as_lora_tuple(self) -> tuple[str, int, str]:
        return self.name, int(self.adapter_id), str(self.path)


@dataclass(frozen=True)
class SampleCall:
    prompt: str
    sampling: dict[str, Any] = field(default_factory=dict)
    adapter: AdapterRef | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = "runtime.model"


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str = "stop"
    token_ids: tuple[int, ...] = ()


__all__ = ["AdapterRef", "Completion", "SampleCall"]
