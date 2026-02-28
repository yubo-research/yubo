"""Types for optimizer module."""

from dataclasses import dataclass
from typing import NamedTuple

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


@dataclass(frozen=True)
class ReturnSummary:
    ret_eval: float
    y_best_s: str
    ret_best_s: str
    ret_eval_s: str
