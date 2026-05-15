from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from llm.console_types import ConsoleEvent


class ConsoleLogFiles:
    def __init__(self, root: Path | None, session_id: str) -> None:
        self.root = root
        self.session_id = session_id
        self.session_dir: Path | None = None
        self.event_log: TextIO | None = None
        self.combined_log: TextIO | None = None
        self.channel_logs: dict[str, TextIO | None] = {"train": None, "inference": None, "diagnostics": None}

    def open(self) -> None:
        if self.root is None:
            return
        self.session_dir = self.root / "console" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = open(self.session_dir / "events.jsonl", "a", encoding="utf-8")
        self.combined_log = open(self.session_dir / "combined.log", "a", encoding="utf-8")
        self.channel_logs["train"] = open(self.session_dir / "train.log", "a", encoding="utf-8")
        self.channel_logs["inference"] = open(self.session_dir / "inference.log", "a", encoding="utf-8")
        self.channel_logs["diagnostics"] = open(self.session_dir / "diagnostics.log", "a", encoding="utf-8")

    def close(self) -> None:
        for handle in [self.event_log, self.combined_log, *self.channel_logs.values()]:
            if handle is not None:
                handle.close()
        self.event_log = None
        self.combined_log = None
        self.channel_logs = {"train": None, "inference": None, "diagnostics": None}

    def flush(self) -> None:
        for handle in [self.event_log, self.combined_log, *self.channel_logs.values()]:
            if handle is not None:
                handle.flush()

    def record_event(self, event: ConsoleEvent) -> None:
        if self.event_log is None:
            return
        self.event_log.write(json.dumps(asdict(event), sort_keys=True, default=str) + "\n")
        self.event_log.flush()

    def write_channel(self, channel: str, line: str) -> None:
        handle = self.channel_logs.get(channel)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()
        if self.combined_log is not None:
            ts = datetime.now(timezone.utc).isoformat()
            self.combined_log.write(f"{ts} {channel:11s} {line}\n")
            self.combined_log.flush()


__all__ = ["ConsoleLogFiles"]
