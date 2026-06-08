from __future__ import annotations

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from llm.console_dashboard_render import (
    diagnostics_renderable,
    footer_renderable,
    header_renderable,
    optimizer_renderable,
    optimizer_state,
    rollout_renderable,
    score_renderable,
    trace_records,
)
from llm.console_observer import SplitConsoleObserver


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
        records = trace_records(self._observer.session_log_dir)
        if records:
            self._follow_latest = False
            self._selected_index = min(len(records) - 1, self._selected_index + 1)
            self._refresh_view()

    def action_up(self) -> None:
        records = trace_records(self._observer.session_log_dir)
        if records:
            self._follow_latest = False
            self._selected_index = max(0, self._selected_index - 1)
            self._refresh_view()

    def action_tab(self) -> None:
        records = trace_records(self._observer.session_log_dir)
        if records:
            self._follow_latest = True
            self._selected_index = len(records) - 1
            self._refresh_view()

    def _refresh_view(self) -> None:
        records = trace_records(self._observer.session_log_dir)
        if records:
            self._selected_index = len(records) - 1 if self._follow_latest else min(max(0, self._selected_index), len(records) - 1)
        state = optimizer_state(self._observer.session_log_dir)
        self.query_one("#dash-header", Static).update(header_renderable(self._observer, state, records))
        self.query_one("#dash-optimizer-body", Static).update(optimizer_renderable(state))
        self.query_one("#dash-rollout-body", Static).update(rollout_renderable(records, self._selected_index))
        self.query_one("#dash-score-body", Static).update(score_renderable(self._observer, state, records, self._selected_index))
        self.query_one("#dash-diag-body", Static).update(diagnostics_renderable(self._observer))
        self.query_one("#dash-footer", Static).update(footer_renderable())

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


__all__ = ["ConsoleDashboard", "run_console_dashboard"]
