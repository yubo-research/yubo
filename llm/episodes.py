from __future__ import annotations

import time

from llm.episode_runtime import Episode, RuntimeConfig
from llm.episode_types import Case, Signal, Turn


def failure_signal(case: Case, *, status: str, error: BaseException | str, reward: float = 0.0) -> Signal:
    return Signal(
        reward=float(reward),
        status=str(status),
        turns=(Turn(kind="system", text=str(error), name="error"),),
        metrics={"case_id": case.id},
        error=str(error),
    )


def signal_log(case: Case, signal: Signal) -> str:
    chunks = [
        f"PROMPT: {case.prompt}",
        f"REWARD: {float(signal.reward)}",
        f"STATUS: {signal.status}",
        "TRAJECTORY:",
    ]
    for turn in signal.turns:
        prefix = turn.kind.upper()
        if turn.name:
            prefix += f" [{turn.name}]"
        chunks.append(f"{prefix}: {turn.text}")
    if signal.error:
        chunks.append(f"ERROR: {signal.error}")
    return "\n".join(chunks)


def summarize_signals(signals: list[Signal]) -> dict[str, float]:
    if not signals:
        return {}
    total = float(len(signals))
    rewards = [float(signal.reward) for signal in signals]
    latencies = [float(signal.metrics["latency_s"]) for signal in signals if isinstance(signal.metrics.get("latency_s"), int | float)]
    turns = [len(signal.turns) for signal in signals]
    return {
        "reward_mean": sum(rewards) / total,
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "turns_mean": sum(turns) / total,
        "failure_rate": sum(1.0 for signal in signals if signal.status not in {"ok", "wrong"}) / total,
        "timeout_rate": sum(1.0 for signal in signals if signal.status == "timeout") / total,
    }


def monotonic_s() -> float:
    return time.perf_counter()


__all__ = [
    "Case",
    "Episode",
    "RuntimeConfig",
    "Signal",
    "Turn",
    "failure_signal",
    "monotonic_s",
    "signal_log",
    "summarize_signals",
]
