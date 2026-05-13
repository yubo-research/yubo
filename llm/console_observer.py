"""
Architectural blueprint for a unified, experiment-agnostic console observer.
Inspired by the 'verifiers' model, this observer pattern allows for non-invasive
telemetry and monitoring of agent loops.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConsoleObserver(Protocol):
    """
    Observer interface for monitoring agent loops.
    Can be attached to verifiers-like loops to capture trajectory and progress.
    """

    async def on_step(self, turn_idx: int, step_data: dict[str, Any]) -> None:
        """Called after each agent-environment interaction turn."""
        ...

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Called when a tool is triggered."""
        ...

    async def on_reward(self, reward: float, metrics: dict[str, Any]) -> None:
        """Called when a reward is issued by the rubric."""
        ...


class UnifiedConsoleManager:
    """
    Manages multiple observers for a running experiment loop.
    Can be used to attach logging, visualization, and progress-tracking.
    """

    def __init__(self):
        self.observers: list[ConsoleObserver] = []

    def attach(self, observer: ConsoleObserver):
        self.observers.append(observer)

    async def broadcast_step(self, turn_idx: int, step_data: dict[str, Any]):
        for obs in self.observers:
            await obs.on_step(turn_idx, step_data)

    async def broadcast_reward(self, reward: float, metrics: dict[str, Any]):
        for obs in self.observers:
            await obs.on_reward(reward, metrics)


class TerminalConsoleObserver:
    """Simple observer that prints structured agent progress to the terminal."""

    async def on_step(self, turn_idx: int, step_data: dict[str, Any]) -> None:
        role = step_data.get("role", "")
        content = step_data.get("content", "")
        if role == "assistant":
            # Look for "Thought:" block
            match = re.search(r"Thought:(.*?)(?=Call Tool|```|$)", content, re.DOTALL)
            if match:
                thought = match.group(1).strip()
                # Print truncated thought
                print(f"  [Turn {turn_idx}] Assistant Thinking: {thought[:120]}...")
        elif role == "tool":
            name = step_data.get("name", "tool")
            output = step_data.get("output", step_data.get("content", ""))
            # Print a snippet of the tool output
            lines = str(output).splitlines()
            snippet = "\n".join(lines[:3])
            if len(lines) > 3:
                snippet += "\n..."
            print(f"  [Turn {turn_idx}] Tool [{name}] output:\n{snippet}")

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        print(f"  >>> Calling Tool: {tool_name}")

    async def on_reward(self, reward: float, metrics: dict[str, Any]) -> None:
        status = metrics.get("status", "unknown")
        print(f"  [Final Reward] {reward:.4f} (Status: {status})")


# Blueprint usage in a verifiers-like loop:
#
# rlm = RLM(env=env, client=client)
# console = UnifiedConsoleManager()
# console.attach(RealTimeVisualizer())
#
# async def rollout(payload):
#     state = await rlm.rollout(payload)
#     await console.broadcast_reward(state.get("reward"), state.get("metrics"))
#     return state
"""
This structure allows us to monitor the loop's 'ambition' and state transitions
without changing how the RL loop itself operates.
"""
