from __future__ import annotations

from enum import StrEnum


class TaskMode(StrEnum):
    SCORE = "score"
    ROLLOUT = "rollout"


__all__ = ["TaskMode"]
