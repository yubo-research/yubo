from __future__ import annotations

from typing import Any, ClassVar, Protocol

from llm.task_modes import TaskMode


class LLMTask(Protocol):
    execution_mode: ClassVar[TaskMode]

    def get_batch(self) -> tuple[list[str], list[Any]]: ...


__all__ = ["LLMTask"]
