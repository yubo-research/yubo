from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from llm.task_protocol_core import LLMTask


class BatchScoringTask(LLMTask, Protocol):
    def score(
        self,
        generations: list[str],
        truncateds: list[bool],
        answer: Any,
        *,
        pass_at_k: bool = False,
    ) -> tuple[float, tuple[Any, ...], np.ndarray]: ...


__all__ = ["BatchScoringTask"]
