"""Append-only console observer for UHD/LLM runs."""

from __future__ import annotations

import os
import shutil
import sys
import threading
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from llm.console_log_files import ConsoleLogFiles
from llm.console_logging import ConsoleLoggingContext
from llm.console_pane import PaneState
from llm.console_tee import tee_stdout_to_exp
from llm.console_text import (
    channel_for_step,
    classify_console_line,
    clean_text,
    is_attention_diagnostic,
)
from llm.console_types import (
    _ACTIVE_OBSERVER,
    ConsoleEvent,
    ConsoleObserver,
    active_console_observer,
    use_console_observer,
)


class UnifiedConsoleManager:
    """Fan-out bus for typed console events."""

    def __init__(self) -> None:
        self.observers: list[ConsoleObserver] = []
        active = active_console_observer()
        if active is not None:
            self.attach(active)

    def attach(self, observer: ConsoleObserver) -> None:
        if observer not in self.observers:
            self.observers.append(observer)

    async def broadcast_step(self, turn_idx: int, step_data: dict[str, Any]) -> None:
        await self.broadcast_event(
            ConsoleEvent(
                kind="step",
                channel=channel_for_step(step_data),
                payload={"turn_idx": turn_idx, **step_data},
            )
        )

    async def broadcast_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        await self.broadcast_event(
            ConsoleEvent(
                kind="tool_call",
                channel="inference",
                payload={"tool_name": tool_name, "args": args},
            )
        )

    async def broadcast_reward(self, reward: float, metrics: dict[str, Any]) -> None:
        await self.broadcast_event(
            ConsoleEvent(
                kind="reward",
                channel="inference",
                payload={"reward": reward, **metrics},
            )
        )

    async def broadcast_event(self, event: ConsoleEvent) -> None:
        for observer in list(self.observers):
            on_event = getattr(observer, "on_event", None)
            if callable(on_event):
                await on_event(event)
            elif event.kind == "step":
                await observer.on_step(int(event.payload.get("turn_idx", -1)), event.payload)
            elif event.kind == "reward":
                await observer.on_reward(float(event.payload.get("reward", 0.0)), event.payload)


class SplitConsoleObserver:
    """Append-only split console plus persistent per-channel logs."""

    def __init__(
        self,
        *,
        max_lines: int = 2000,
        stream: TextIO | None = None,
        log_dir: str | os.PathLike[str] | None = None,
        enable_tui: bool = True,
        diagnostics_to_stream: bool = False,
    ) -> None:
        self.max_lines = int(max_lines)
        self._stream = stream if stream is not None else (sys.stdout if enable_tui else None)
        self._diagnostics_to_stream = bool(diagnostics_to_stream)
        self._panes = _new_panes(self.max_lines)
        self._active_pane = "train"
        self._lock = threading.RLock()
        self._status = "starting"
        self._active_token = None
        self._logging_context: ConsoleLoggingContext | None = None
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        self._logs = ConsoleLogFiles(Path(log_dir) if log_dir is not None else None, session_id)

    def __enter__(self):
        self._logs.open()
        self._active_token = _ACTIVE_OBSERVER.set(self)
        self._logging_context = ConsoleLoggingContext(self)
        self._logging_context.__enter__()
        _write_banner(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def session_log_dir(self) -> Path | None:
        return self._logs.session_dir

    @property
    def owns_terminal_input(self) -> bool:
        return False

    def close(self) -> None:
        _close_observer(self)

    async def on_step(self, turn_idx: int, step_data: dict[str, Any]) -> None:
        await _on_step(self, turn_idx, step_data)

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        self.append_inference(
            f"tool call: {tool_name} args={args}",
            kind="tool_call",
            payload={"tool_name": tool_name, "args": args},
        )

    async def on_reward(self, reward: float, metrics: dict[str, Any]) -> None:
        status = metrics.get("status", "unknown")
        self.append_inference(
            f"REWARD: {reward:.4f} status={status}",
            kind="reward",
            payload={"reward": reward, **metrics},
        )

    async def on_event(self, event: ConsoleEvent) -> None:
        await _on_event(self, event)

    def append_train(
        self,
        text: str,
        *,
        kind: str = "line",
        payload: dict[str, Any] | None = None,
        record: bool = True,
    ) -> None:
        _append(self, "train", text, kind=kind, payload=payload, record=record)

    def append_inference(
        self,
        text: str,
        *,
        kind: str = "line",
        payload: dict[str, Any] | None = None,
        record: bool = True,
    ) -> None:
        _append(self, "inference", text, kind=kind, payload=payload, record=record)

    def append_diagnostics(
        self,
        text: str,
        *,
        kind: str = "line",
        payload: dict[str, Any] | None = None,
        record: bool = True,
    ) -> None:
        _append(self, "diagnostics", text, kind=kind, payload=payload, record=record)

    def route_line(self, line: str) -> None:
        _route_line(self, line)

    def flush(self) -> None:
        _flush(self)

    @contextmanager
    def output_to(self, stream: TextIO):
        old_stream = self._stream
        self._stream = stream
        try:
            yield
        finally:
            self._stream = old_stream


class TerminalConsoleObserver(SplitConsoleObserver):
    pass


def _new_panes(max_lines: int) -> dict[str, PaneState]:
    return {
        "train": PaneState("train", "train", deque(maxlen=max_lines)),
        "inference": PaneState("inference", "model", deque(maxlen=max_lines)),
        "diagnostics": PaneState("diagnostics", "diag", deque(maxlen=max_lines)),
    }


def _append_exp(
    observer: SplitConsoleObserver,
    text: str,
    *,
    kind: str = "line",
    payload: dict[str, Any] | None = None,
    record: bool = True,
) -> None:
    observer.append_train(text, kind=kind, payload=payload, record=record)


def _append_model(
    observer: SplitConsoleObserver,
    text: str,
    *,
    kind: str = "line",
    payload: dict[str, Any] | None = None,
    record: bool = True,
) -> None:
    observer.append_inference(text, kind=kind, payload=payload, record=record)


def _close_observer(observer: SplitConsoleObserver) -> None:
    if observer._logging_context is not None:
        observer._logging_context.__exit__(None, None, None)
        observer._logging_context = None
    if observer._active_token is not None:
        _ACTIVE_OBSERVER.reset(observer._active_token)
        observer._active_token = None
    observer.flush()
    observer._logs.close()


async def _on_step(observer: SplitConsoleObserver, turn_idx: int, step_data: dict[str, Any]) -> None:
    role = str(step_data.get("role", ""))
    if role == "assistant":
        observer.append_inference(
            f"[turn {turn_idx}] assistant",
            kind="assistant",
            payload={"turn_idx": turn_idx, **step_data},
        )
        observer.append_inference(str(step_data.get("content", "")), record=False)
        return
    if role == "tool":
        name = str(step_data.get("name", "tool"))
        observer.append_inference(
            f"[turn {turn_idx}] tool[{name}]",
            kind="tool",
            payload={"turn_idx": turn_idx, **step_data},
        )
        observer.append_inference(str(step_data.get("output", step_data.get("content", ""))), record=False)
        return
    content = str(step_data.get("content", step_data.get("output", "")))
    observer.append_train(
        f"[turn {turn_idx}] {role}: {content}",
        kind="step",
        payload={"turn_idx": turn_idx, **step_data},
    )


async def _on_event(observer: SplitConsoleObserver, event: ConsoleEvent) -> None:
    if event.kind == "step":
        await observer.on_step(int(event.payload.get("turn_idx", -1)), event.payload)
        return
    if event.kind == "reward":
        await observer.on_reward(float(event.payload.get("reward", 0.0)), event.payload)
        return
    text = str(event.payload.get("line", event.payload.get("content", event.kind)))
    _append_by_channel(observer, event.channel, text, kind=event.kind, payload=event.payload)


def _append_by_channel(
    observer: SplitConsoleObserver,
    channel: str,
    text: str,
    *,
    kind: str,
    payload: dict[str, Any],
) -> None:
    if channel == "inference":
        observer.append_inference(text, kind=kind, payload=payload)
    elif channel == "diagnostics":
        observer.append_diagnostics(text, kind=kind, payload=payload)
    else:
        observer.append_train(text, kind=kind, payload=payload)


def _append(
    observer: SplitConsoleObserver,
    channel: str,
    text: str,
    *,
    kind: str,
    payload: dict[str, Any] | None,
    record: bool,
) -> None:
    clean = clean_text(str(text))
    lines = clean.splitlines() or [""]
    with observer._lock:
        if record:
            observer._logs.record_event(ConsoleEvent(kind=kind, channel=channel, payload=payload or {"text": clean}))
        for line in lines:
            observer._panes.get(channel, observer._panes["train"]).append(line)
            observer._logs.write_channel(channel, line)
            _update_status(observer, channel, line)
        _write_stream(observer, channel, lines)


def _route_line(observer: SplitConsoleObserver, line: str) -> None:
    if not line:
        return
    channel = classify_console_line(line)
    _append_by_channel(observer, channel, line, kind="stdout", payload={"line": line})


def _flush(observer: SplitConsoleObserver) -> None:
    observer._logs.flush()
    stream = observer._stream
    if stream is not None:
        stream.flush()


def _update_status(observer: SplitConsoleObserver, channel: str, line: str) -> None:
    if channel == "diagnostics":
        if "error" in line.lower() or "warning" in line.lower():
            observer._status = f"diagnostics: {line[:160]}"
        return
    if line.startswith(("EVAL:", "ITER:", "REWARD:", "RESULT:", "LLM_EGGROLL:")):
        observer._status = line[:200]


def _write_stream(observer: SplitConsoleObserver, channel: str, lines: list[str]) -> None:
    stream = observer._stream
    if stream is None:
        return
    if channel == "diagnostics" and not observer._diagnostics_to_stream:
        if not any(is_attention_diagnostic(line) for line in lines):
            return
    prefix = _STREAM_PREFIX[channel]
    for line in lines:
        stream.write(f"{prefix} | {line}\n")
    stream.flush()


def _write_banner(observer: SplitConsoleObserver) -> None:
    stream = observer._stream
    if stream is None or observer.session_log_dir is None:
        return
    stream.write(f"console | logs: {observer.session_log_dir}\n")
    stream.flush()


def _render_channel(observer: SplitConsoleObserver, channel: str) -> str:
    with observer._lock:
        lines = observer._panes.get(channel, observer._panes["train"]).visible_lines(height=_pane_height())
    return "\n".join(lines)


def _pane_title(observer: SplitConsoleObserver, channel: str) -> str:
    with observer._lock:
        pane = observer._panes.get(channel, observer._panes["train"])
        mode = "tail" if pane.follow_tail else f"frozen +{pane.unseen_count}"
    return f"{pane.title} [{mode}]"


def _cycle_focus(observer: SplitConsoleObserver) -> None:
    with observer._lock:
        observer._active_pane = "inference" if observer._active_pane == "train" else "train"


def _scroll_active(observer: SplitConsoleObserver, delta: int) -> None:
    with observer._lock:
        observer._panes[observer._active_pane].scroll(delta, height=_pane_height())


def _follow_active(observer: SplitConsoleObserver) -> None:
    with observer._lock:
        observer._panes[observer._active_pane].follow()


def _clear_active(observer: SplitConsoleObserver) -> None:
    with observer._lock:
        observer._panes[observer._active_pane].clear()


def _pane_height() -> int:
    rows = shutil.get_terminal_size(fallback=(120, 32)).lines
    return max(5, rows - 4)


_STREAM_PREFIX = {
    "train": "train",
    "inference": "model",
    "diagnostics": "diag ",
}

SplitConsoleObserver._render_channel = _render_channel
SplitConsoleObserver._pane_title = _pane_title
SplitConsoleObserver._cycle_focus = _cycle_focus
SplitConsoleObserver._scroll_active = _scroll_active
SplitConsoleObserver._follow_active = _follow_active
SplitConsoleObserver._clear_active = _clear_active
SplitConsoleObserver._pane_height = lambda self: _pane_height()
SplitConsoleObserver.append_exp = _append_exp
SplitConsoleObserver.append_model = _append_model


__all__ = [
    "ConsoleEvent",
    "ConsoleObserver",
    "SplitConsoleObserver",
    "TerminalConsoleObserver",
    "UnifiedConsoleManager",
    "active_console_observer",
    "tee_stdout_to_exp",
    "use_console_observer",
]
