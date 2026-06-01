from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalPoint:
    index: int
    values: dict[str, str]


@dataclass
class OptimizerState:
    config: dict[str, str]
    text_runtime: dict[str, str]
    evals: list[EvalPoint]


@dataclass
class TraceRecord:
    title: str
    summary: str
    body: list[str]
    metrics: dict[str, str]


__all__ = ["EvalPoint", "OptimizerState", "TraceRecord"]
