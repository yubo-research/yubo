from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm.console_dashboard_types import EvalPoint, OptimizerState, TraceRecord
from llm.console_observer import SplitConsoleObserver

_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)")


def header_renderable(observer: SplitConsoleObserver, state: OptimizerState, records: list[TraceRecord]) -> Group:
    latest = state.evals[-1].values if state.evals else {}
    latest_record = records[-1] if records else None
    headline = latest_record.summary if latest_record is not None else headline_status(observer.session_log_dir)
    text = Text()
    text.append("UHD agent optimizer", style="bold #d6a84f")
    text.append("  ")
    text.append(_kv("env", state.config.get("env_tag", "?")), style="dim")
    text.append("  ")
    text.append(_kv("optimizer", state.config.get("optimizer", "?")), style="dim")
    text.append("  ")
    text.append(_kv("iter", latest.get("i_iter", latest.get("step", "?"))), style="dim")
    text.append("  ")
    text.append(headline or "waiting for first rollout", style="bold #78c77a")
    return Group(Panel(text, border_style="#d6a84f", padding=(0, 1), box=box.ROUNDED))


def optimizer_renderable(state: OptimizerState) -> Group:
    body = Group(_optimizer_summary(state), Text(""), _trend_table(state))
    return Group(Panel(body, title="Optimizer", border_style="#d6a84f", padding=(0, 1), box=box.ROUNDED))


def rollout_renderable(records: list[TraceRecord], selected_index: int) -> Group:
    if not records:
        waiting = Text("Waiting for the first model rollout.", style="dim")
        waiting.append("\n\nThe optimizer is still initializing runtime or sampling the first candidate.", style="dim")
        return Group(Panel(waiting, title="Rollout", border_style="#4a3b27", padding=(1, 2), box=box.ROUNDED))
    selected = records[min(selected_index, len(records) - 1)]
    title = f"Rollout {min(selected_index, len(records) - 1) + 1}/{len(records)}  {selected.title}"
    return Group(Panel(transcript_text(selected), title=title, border_style="#78c77a", padding=(1, 2), box=box.ROUNDED))


def score_renderable(
    observer: SplitConsoleObserver,
    state: OptimizerState,
    records: list[TraceRecord],
    selected_index: int,
) -> Group:
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(ratio=1)
    selected = records[min(selected_index, len(records) - 1)] if records else None
    summary = _key_values(selected.summary) if selected is not None else {}
    latest_eval = state.evals[-1].values if state.evals else {}
    table.add_row("score", summary.get("reward", "?"))
    table.add_row("status", summary.get("status", observer._status))
    table.add_row("case", selected.title if selected is not None else "?")
    table.add_row("latency", _metric(selected, "latency_s"))
    table.add_row("turns", _metric(selected, "turns"))
    table.add_row("rollout", str(len(records)))
    table.add_row("eval mu", latest_eval.get("mu", "?"))
    table.add_row("eval best", latest_eval.get("y_best", "?"))
    table.add_row("session", _short(str(observer.session_log_dir or "memory"), 42))
    return Group(Panel(table, title="Scoring", border_style="#6fc3df", padding=(0, 1), box=box.ROUNDED))


def diagnostics_renderable(observer: SplitConsoleObserver) -> Group:
    session_dir = observer.session_log_dir
    lines = _tail_lines(session_dir / "diagnostics.log", limit=4) if session_dir is not None else []
    text = Text()
    if not lines:
        text.append("No diagnostics.", style="dim")
    else:
        for idx, line in enumerate(lines):
            if idx:
                text.append("\n")
            text.append(_short(line, 220), style="dim")
    return Group(Panel(text, title="Diagnostics", border_style="#4a3b27", padding=(0, 1), box=box.ROUNDED))


def footer_renderable() -> Group:
    text = Text()
    text.append("up/down", style="bold #6fc3df")
    text.append(" select rollout   ", style="dim")
    text.append("tab", style="bold #6fc3df")
    text.append(" follow latest   ", style="dim")
    text.append("q", style="bold #6fc3df")
    text.append(" quit", style="dim")
    return Group(Panel(text, border_style="#4a3b27", padding=(0, 1), box=box.ROUNDED))


def optimizer_state(session_dir: Path | None) -> OptimizerState:
    if session_dir is None:
        return OptimizerState(config={}, text_runtime={}, evals=[])
    config: dict[str, str] = {}
    text_runtime: dict[str, str] = {}
    evals: list[EvalPoint] = []
    for line in _tail_lines(session_dir / "train.log", limit=600):
        if line.startswith("UHD-Vector:"):
            config.update(_key_values(line))
        elif line.startswith("UHD-Text:"):
            text_runtime.update(_key_values(line))
        elif line.startswith("EVAL:"):
            evals.append(EvalPoint(index=len(evals), values=_key_values(line)))
    return OptimizerState(config=config, text_runtime=text_runtime, evals=evals)


def trace_records(session_dir: Path | None) -> list[TraceRecord]:
    if session_dir is None:
        return []
    lines = _tail_lines(session_dir / "inference.log", limit=4000)
    records: list[TraceRecord] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("CASE: ") and current:
            parsed = _parse_trace_block(current)
            if parsed is not None:
                records.append(parsed)
            current = []
        if line.strip():
            current.append(line)
    if current:
        parsed = _parse_trace_block(current)
        if parsed is not None:
            records.append(parsed)
    return records


def headline_status(session_dir: Path | None) -> str | None:
    records = trace_records(session_dir)
    if records:
        return records[-1].summary
    state = optimizer_state(session_dir)
    if state.evals:
        values = state.evals[-1].values
        return f"mu={values.get('mu', '?')} best={values.get('y_best', '?')}"
    return None


def transcript_text(record: TraceRecord) -> Text:
    out = Text()
    ctx = _TranscriptContext()
    for raw in record.body:
        line = raw.strip()
        if not line:
            continue
        _dispatch_transcript_line(out, ctx, line)
    if ctx.prompt_buffer:
        for prompt_line in ctx.prompt_buffer[-8:]:
            _append_plain(out, prompt_line, "dim")
    return out


class _TranscriptContext:
    __slots__ = ("in_trace", "prompt_buffer")

    def __init__(self) -> None:
        self.in_trace = False
        self.prompt_buffer: list[str] = []


def _dispatch_transcript_line(out: Text, ctx: _TranscriptContext, line: str) -> None:
    if line.startswith("TRACE:"):
        _handle_trace_marker(out, ctx)
        return
    if not ctx.in_trace:
        handler = _PROMPT_HANDLERS.get(line.partition(":")[0] + ":", _handle_prompt_body)
        handler(out, ctx, line)
        return
    if line.startswith("- "):
        return
    if line.startswith("TOOL"):
        _handle_tool_line(out, line)
        return
    handler = _TRACE_HANDLERS.get(line.partition(":")[0] + ":", _handle_trace_body)
    handler(out, ctx, line)


def _handle_trace_marker(out: Text, ctx: _TranscriptContext) -> None:
    if ctx.prompt_buffer:
        for prompt_line in ctx.prompt_buffer[-8:]:
            _append_plain(out, prompt_line, "dim")
        ctx.prompt_buffer = []
    ctx.in_trace = True
    _append_section(out, "trajectory")


def _handle_prompt_prefix(out: Text, ctx: _TranscriptContext, line: str) -> None:
    prefix, _, value = line.partition(":")
    if prefix == "CASE":
        _append_line(out, "case", value.strip(), "bold #6fc3df")
    elif prefix == "PROMPT":
        if value.strip():
            _append_line(out, "prompt", value.strip(), "bold #eee7da")
        else:
            _append_section(out, "prompt")
    elif prefix == "SUMMARY":
        _append_line(out, "score", value.strip(), "bold #78c77a")
    elif prefix == "METRICS":
        _append_line(out, "metrics", _short(value.strip(), 160), "dim")


def _handle_prompt_body(out: Text, ctx: _TranscriptContext, line: str) -> None:
    if line.startswith("- "):
        ctx.prompt_buffer.append(line.removeprefix("- ").lower() + ":")
    else:
        ctx.prompt_buffer.append(line)


def _handle_tool_line(out: Text, line: str) -> None:
    label, _, value = line.partition(":")
    _append_line(out, label.lower(), value.strip(), "#d6a84f")


def _handle_trace_prefix(out: Text, ctx: _TranscriptContext, line: str) -> None:
    prefix, _, value = line.partition(":")
    if prefix == "USER":
        _append_line(out, "user", value.strip(), "bold #eee7da")
    elif prefix == "ASSISTANT":
        _append_line(out, "assistant", value.strip(), "bold #78c77a")
    elif prefix == "ASSISTANT_REASONING":
        _append_line(out, "reasoning", value.strip(), "dim")
    elif prefix == "ASSISTANT_TOOL_CALL":
        _append_line(out, "tool call", _format_tool_call_line(value.strip()), "bold #d6a84f")
    elif prefix == "ERROR":
        _append_line(out, "error", value.strip(), "bold #d88949")


def _handle_trace_body(out: Text, ctx: _TranscriptContext, line: str) -> None:
    _append_plain(out, line, "dim")


_PROMPT_HANDLERS: dict[str, Callable[[Text, _TranscriptContext, str], None]] = {
    "CASE:": _handle_prompt_prefix,
    "PROMPT:": _handle_prompt_prefix,
    "SUMMARY:": _handle_prompt_prefix,
    "METRICS:": _handle_prompt_prefix,
}
_TRACE_HANDLERS: dict[str, Callable[[Text, _TranscriptContext, str], None]] = {
    "USER:": _handle_trace_prefix,
    "ASSISTANT:": _handle_trace_prefix,
    "ASSISTANT_REASONING:": _handle_trace_prefix,
    "ASSISTANT_TOOL_CALL:": _handle_trace_prefix,
    "ERROR:": _handle_trace_prefix,
}


def _parse_trace_block(lines: list[str]) -> TraceRecord | None:
    if not lines:
        return None
    title = "rollout"
    summary = ""
    metrics: dict[str, str] = {}
    for line in lines:
        if line.startswith("CASE: "):
            title = line.removeprefix("CASE: ").strip()
        elif line.startswith("SUMMARY: "):
            summary = line.removeprefix("SUMMARY: ").strip()
        elif line.startswith("METRICS: "):
            metrics = _metrics_from_line(line)
    if not summary:
        summary = "pending"
    return TraceRecord(title=title, summary=summary, body=lines[-80:], metrics=metrics)


def _optimizer_summary(state: OptimizerState) -> Table:
    latest = state.evals[-1].values if state.evals else {}
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(ratio=1)
    table.add_row("model", _short(state.text_runtime.get("model", "?"), 30))
    table.add_row("dim", state.config.get("dim", "?"))
    table.add_row("sigma", latest.get("sigma", state.config.get("perturb", "?")))
    table.add_row("mu", latest.get("mu", "?"))
    table.add_row("se", latest.get("se", "?"))
    table.add_row("best", latest.get("y_best", "?"))
    table.add_row("trend", _sparkline([_float_or_none(point.values.get("mu")) for point in state.evals[-24:]]))
    return table


def _trend_table(state: OptimizerState) -> Table:
    table = Table(show_header=True, header_style="dim", expand=True, box=box.SIMPLE)
    table.add_column("iter", no_wrap=True)
    table.add_column("mu", justify="right")
    table.add_column("best", justify="right")
    for point in state.evals[-8:]:
        values = point.values
        table.add_row(values.get("i_iter", values.get("step", str(point.index))), values.get("mu", ""), values.get("y_best", ""))
    if not state.evals:
        table.add_row("-", "-", "-")
    return table


def _metrics_from_line(line: str) -> dict[str, str]:
    payload = line.removeprefix("METRICS: ").strip()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return {"metrics": payload}
    if not isinstance(raw, dict):
        return {"metrics": str(raw)}
    return {str(key): _short(str(value), 72) for key, value in raw.items() if value is not None}


def _metric(record: TraceRecord | None, key: str) -> str:
    if record is None:
        return "?"
    return record.metrics.get(key, "?")


def _sparkline(values: list[float | None]) -> str:
    nums = [value for value in values if value is not None and math.isfinite(value)]
    if not nums:
        return "-"
    lo = min(nums)
    hi = max(nums)
    if lo == hi:
        return "=" * min(24, len(nums))
    chars = "▁▂▃▄▅▆▇█"
    out = []
    for value in nums[-24:]:
        idx = int(round(((value - lo) / (hi - lo)) * (len(chars) - 1)))
        out.append(chars[max(0, min(len(chars) - 1, idx))])
    return "".join(out)


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_tool_call_line(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return _short(value, 180)
    if not isinstance(parsed, dict):
        return _short(str(parsed), 180)
    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if name and arguments is not None:
        return _short(f"{name} {arguments}", 180)
    return _short(json.dumps(parsed, sort_keys=True), 180)


def _append_section(out: Text, label: str) -> None:
    if len(out):
        out.append("\n\n")
    out.append(label.upper(), style="bold #d6a84f")


def _append_line(out: Text, label: str, value: str, style: str) -> None:
    if len(out):
        out.append("\n")
    out.append(f"{label:>10} ", style="dim")
    out.append(value or "<empty>", style=style)


def _append_plain(out: Text, value: str, style: str) -> None:
    if len(out):
        out.append("\n")
    out.append(" " * 11)
    out.append(value, style=style)


def _tail_lines(path: Path, *, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []


def _key_values(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _KV_RE.finditer(line)}


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _kv(key: str, value: str) -> str:
    return f"{key}={value}"


__all__ = [
    "diagnostics_renderable",
    "footer_renderable",
    "headline_status",
    "header_renderable",
    "optimizer_renderable",
    "optimizer_state",
    "rollout_renderable",
    "score_renderable",
    "trace_records",
    "transcript_text",
]
