"""Types for optimizer module."""

from dataclasses import dataclass
from typing import Any, NamedTuple

from .datum import Datum


class IterateResult(NamedTuple):
    data: list[Datum]
    dt_prop: float
    dt_eval: float


@dataclass
class TraceEntry:
    rreturn: float
    rreturn_decision: float
    dt_prop: float
    dt_eval: float
    env_steps_iter: int = 0
    env_steps_total: int = 0


@dataclass(frozen=True)
class ReturnSummary:
    ret_eval: float
    y_best_s: str
    ret_best_s: str
    ret_eval_s: str


@dataclass(frozen=True)
class VideoReplaySpec:
    policy: Any
    noise_seed: int
    iter_index: int
    raw_return: float
    estimated_return: float
