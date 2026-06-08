from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class PaneState:
    channel: str
    title: str
    lines: deque[str]
    scroll_offset: int = 0
    follow_tail: bool = True
    unseen_count: int = 0

    def append(self, line: str) -> None:
        self.lines.append(line)
        if not self.follow_tail:
            self.unseen_count += 1

    def clear(self) -> None:
        self.lines.clear()
        self.scroll_offset = 0
        self.follow_tail = True
        self.unseen_count = 0

    def follow(self) -> None:
        self.follow_tail = True
        self.unseen_count = 0

    def scroll(self, delta: int, *, height: int) -> None:
        max_start = self.max_start(height=height)
        if self.follow_tail:
            self.scroll_offset = max_start
            self.follow_tail = False
        self.scroll_offset = min(max(0, self.scroll_offset + delta), max_start)
        if self.scroll_offset >= max_start:
            self.follow()

    def visible_lines(self, *, height: int) -> list[str]:
        lines = list(self.lines)
        if not lines:
            return []
        if self.follow_tail:
            start = self.max_start(height=height)
        else:
            max_start = self.max_start(height=height)
            self.scroll_offset = min(max(0, self.scroll_offset), max_start)
            start = self.scroll_offset
        return lines[start : start + height]

    def max_start(self, *, height: int) -> int:
        return max(0, len(self.lines) - max(1, height))


__all__ = ["PaneState"]
