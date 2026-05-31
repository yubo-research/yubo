from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from llm.console_observer import SplitConsoleObserver


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


class ConsoleDashboard(App[None]):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("up", "up", "Previous"),
        Binding("down", "down", "Next"),
        Binding("tab", "tab", "Latest"),
    ]

    CSS = """
    ConsoleDashboard {
        background: #12100d;
        color: #eee7da;
        layout: vertical;
    }

    #dash-header {
        height: 4;
        padding: 0 1;
        border-bottom: solid #d6a84f;
        background: #17130f;
    }

    #dash-body {
        height: 1fr;
        padding: 0 1;
    }

    #dash-main {
        height: 1fr;
    }

    #dash-optimizer {
        width: 34;
        min-width: 30;
        margin-right: 1;
    }

    #dash-rollout {
        width: 1fr;
        margin-right: 1;
    }

    #dash-score {
        width: 38;
        min-width: 32;
    }

    .dash-pane {
        border: round #4a3b27;
        background: #17130f;
        padding: 0 1;
    }

    #dash-diag {
        height: 7;
        min-height: 6;
        margin-top: 1;
    }

    #dash-footer {
        height: 3;
        padding: 0 1;
        border-top: solid #4a3b27;
        background: #17130f;
    }
    """

    def __init__(
        self,
        observer: SplitConsoleObserver,
        *,
        title: str = "Console Dashboard",
        done_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._observer = observer
        self._done_event = done_event
        self.title = title
        self._selected_index = 0
        self._follow_latest = True

    def compose(self) -> ComposeResult:
        yield Static("", id="dash-header")
        with Vertical(id="dash-body"):
            with Horizontal(id="dash-main"):
                with Vertical(classes="dash-pane", id="dash-optimizer"):
                    yield Static("", id="dash-optimizer-body")
                with Vertical(classes="dash-pane", id="dash-rollout"):
                    yield Static("", id="dash-rollout-body")
                with Vertical(classes="dash-pane", id="dash-score"):
                    yield Static("", id="dash-score-body")
            with Vertical(classes="dash-pane", id="dash-diag"):
                yield Static("", id="dash-diag-body")
        yield Static("", id="dash-footer")

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh_view)
        self.set_interval(0.25, self._maybe_exit)
        self._refresh_view()

    def action_down(self) -> None:
        records = _trace_records(self._observer.session_log_dir)
        if records:
            self._follow_latest = False
            self._selected_index = min(len(records) - 1, self._selected_index + 1)
            self._refresh_view()

    def action_up(self) -> None:
        records = _trace_records(self._observer.session_log_dir)
        if records:
            self._follow_latest = False
            self._selected_index = max(0, self._selected_index - 1)
            self._refresh_view()

    def action_tab(self) -> None:
        records = _trace_records(self._observer.session_log_dir)
        if records:
            self._follow_latest = True
            self._selected_index = len(records) - 1
            self._refresh_view()

    def _refresh_view(self) -> None:
        records = _trace_records(self._observer.session_log_dir)
        if records:
            self._selected_index = len(records) - 1 if self._follow_latest else min(max(0, self._selected_index), len(records) - 1)
        state = _optimizer_state(self._observer.session_log_dir)
        self.query_one("#dash-header", Static).update(_header_renderable(self._observer, state, records))
        self.query_one("#dash-optimizer-body", Static).update(_optimizer_renderable(state))
        self.query_one("#dash-rollout-body", Static).update(_rollout_renderable(records, self._selected_index))
        self.query_one("#dash-score-body", Static).update(_score_renderable(self._observer, state, records, self._selected_index))
        self.query_one("#dash-diag-body", Static).update(_diagnostics_renderable(self._observer))
        self.query_one("#dash-footer", Static).update(_footer_renderable())

    def _maybe_exit(self) -> None:
        if self._done_event is not None and self._done_event.is_set():
            self.exit()


def run_console_dashboard(
    observer: SplitConsoleObserver,
    *,
    title: str = "Console Dashboard",
    done_event: threading.Event | None = None,
) -> None:
    ConsoleDashboard(observer, title=title, done_event=done_event).run()


def _header_renderable(observer: SplitConsoleObserver, state: OptimizerState, records: list[TraceRecord]) -> Group:
    latest = state.evals[-1].values if state.evals else {}
    latest_record = records[-1] if records else None
    headline = latest_record.summary if latest_record is not None else _headline_status(observer.session_log_dir)
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


def _optimizer_renderable(state: OptimizerState) -> Group:
    body = Group(_optimizer_summary(state), Text(""), _trend_table(state))
    return Group(Panel(body, title="Optimizer", border_style="#d6a84f", padding=(0, 1), box=box.ROUNDED))


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


def _rollout_renderable(records: list[TraceRecord], selected_index: int) -> Group:
    if not records:
        waiting = Text("Waiting for the first model rollout.", style="dim")
        waiting.append("\n\nThe optimizer is still initializing runtime or sampling the first candidate.", style="dim")
        return Group(Panel(waiting, title="Rollout", border_style="#4a3b27", padding=(1, 2), box=box.ROUNDED))
    selected = records[min(selected_index, len(records) - 1)]
    title = f"Rollout {min(selected_index, len(records) - 1) + 1}/{len(records)}  {selected.title}"
    return Group(Panel(_transcript_text(selected), title=title, border_style="#78c77a", padding=(1, 2), box=box.ROUNDED))


def _score_renderable(
    observer: SplitConsoleObserver,
    state: OptimizerState,
    records: list[TraceRecord],
    selected_index: int,
) -> Group:
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(ratio=1)
    selected = records[min(selected_index, len(records) - 1)] if records else None
    summary = _summary_fields(selected.summary) if selected is not None else {}
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


def _diagnostics_renderable(observer: SplitConsoleObserver) -> Group:
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


def _footer_renderable() -> Group:
    text = Text()
    text.append("up/down", style="bold #6fc3df")
    text.append(" select rollout   ", style="dim")
    text.append("tab", style="bold #6fc3df")
    text.append(" follow latest   ", style="dim")
    text.append("q", style="bold #6fc3df")
    text.append(" quit", style="dim")
    return Group(Panel(text, border_style="#4a3b27", padding=(0, 1), box=box.ROUNDED))


def _optimizer_state(session_dir: Path | None) -> OptimizerState:
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


def _trace_records(session_dir: Path | None) -> list[TraceRecord]:
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


def _transcript_text(record: TraceRecord) -> Text:
    out = Text()
    in_trace = False
    prompt_buffer: list[str] = []
    for raw in record.body:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("CASE: "):
            _append_line(out, "case", line.removeprefix("CASE: "), "bold #6fc3df")
        elif line.startswith("PROMPT:"):
            value = line.removeprefix("PROMPT:").strip()
            if value:
                _append_line(out, "prompt", value, "bold #eee7da")
            else:
                _append_section(out, "prompt")
        elif line.startswith("SUMMARY: "):
            _append_line(out, "score", line.removeprefix("SUMMARY: "), "bold #78c77a")
        elif line.startswith("METRICS: "):
            _append_line(out, "metrics", _short(line.removeprefix("METRICS: "), 160), "dim")
        elif line.startswith("TRACE:"):
            if prompt_buffer:
                for prompt_line in prompt_buffer[-8:]:
                    _append_plain(out, prompt_line, "dim")
                prompt_buffer = []
            in_trace = True
            _append_section(out, "trajectory")
        elif not in_trace and line.startswith("- "):
            prompt_buffer.append(line.removeprefix("- ").lower() + ":")
        elif not in_trace:
            prompt_buffer.append(line)
        elif line.startswith("- "):
            continue
        elif line.startswith("USER: "):
            _append_line(out, "user", line.removeprefix("USER: "), "bold #eee7da")
        elif line.startswith("ASSISTANT: "):
            _append_line(out, "assistant", line.removeprefix("ASSISTANT: "), "bold #78c77a")
        elif line.startswith("ASSISTANT_REASONING: "):
            _append_line(out, "reasoning", line.removeprefix("ASSISTANT_REASONING: "), "dim")
        elif line.startswith("ASSISTANT_TOOL_CALL: "):
            _append_line(out, "tool call", _format_tool_call_line(line.removeprefix("ASSISTANT_TOOL_CALL: ")), "bold #d6a84f")
        elif line.startswith("TOOL"):
            label, _, value = line.partition(":")
            _append_line(out, label.lower(), value.strip(), "#d6a84f")
        elif line.startswith("ERROR: "):
            _append_line(out, "error", line.removeprefix("ERROR: "), "bold #d88949")
        else:
            _append_plain(out, line, "dim")
    if prompt_buffer:
        for prompt_line in prompt_buffer[-8:]:
            _append_plain(out, prompt_line, "dim")
    return out


def _metrics_from_line(line: str) -> dict[str, str]:
    payload = line.removeprefix("METRICS: ").strip()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return {"metrics": payload}
    if not isinstance(raw, dict):
        return {"metrics": str(raw)}
    return {str(key): _short(str(value), 72) for key, value in raw.items() if value is not None}


def _headline_status(session_dir: Path | None) -> str | None:
    records = _trace_records(session_dir)
    if records:
        return records[-1].summary
    state = _optimizer_state(session_dir)
    if state.evals:
        values = state.evals[-1].values
        return f"mu={values.get('mu', '?')} best={values.get('y_best', '?')}"
    return None


def _key_values(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _KV_RE.finditer(line)}


def _summary_fields(summary: str) -> dict[str, str]:
    return _key_values(summary)


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


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except OSError:
        return 0


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _kv(key: str, value: str) -> str:
    return f"{key}={value}"


_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)")


__all__ = ["ConsoleDashboard", "run_console_dashboard"]
