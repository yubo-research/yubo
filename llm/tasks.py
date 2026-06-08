from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import task_protocols as _protocols
from .tasks_base import (
    BatchScoringTaskMixin,
    RolloutTaskMixin,
    TaskMode,
    extract_model_answer,
    is_rollout_task,
    score_generations,
    task_mode,
)
from .tasks_countdown import (
    CountdownTask,
    countdown_answer_reward,
    countdown_format_reward,
)
from .tasks_factory import build_task
from .tasks_math import MathTask, MathTaskConfig, check_math_correct
from .tasks_static import RandomTask, ZerosTask

if TYPE_CHECKING:
    from .tasks_verifiers import VerifiersTask, VerifiersTaskConfig

# Explicitly re-export protocols
AsyncRolloutTask = _protocols.AsyncRolloutTask
BatchScoringTask = _protocols.BatchScoringTask
LLMTask = _protocols.LLMTask
RolloutTask = _protocols.RolloutTask


def __getattr__(name: str) -> Any:
    if name in ("VerifiersTask", "VerifiersTaskConfig"):
        from . import tasks_verifiers

        return getattr(tasks_verifiers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncRolloutTask",
    "BatchScoringTask",
    "BatchScoringTaskMixin",
    "CountdownTask",
    "LLMTask",
    "MathTask",
    "MathTaskConfig",
    "RandomTask",
    "RolloutTask",
    "RolloutTaskMixin",
    "TaskMode",
    "VerifiersTask",
    "VerifiersTaskConfig",
    "ZerosTask",
    "build_task",
    "check_math_correct",
    "countdown_answer_reward",
    "countdown_format_reward",
    "extract_model_answer",
    "is_rollout_task",
    "score_generations",
    "task_mode",
]
