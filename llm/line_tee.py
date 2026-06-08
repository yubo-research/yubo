from __future__ import annotations

from typing import Callable, TextIO


class LineRoutingTee:
    def __init__(self, stream: TextIO, route_line: Callable[[str], None], *, echo: bool = False) -> None:
        self._stream = stream
        self._route_line = route_line
        self._echo = bool(echo)
        self._buf = ""

    def write(self, data):
        if self._echo:
            self._stream.write(data)
            self._stream.flush()
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._route_line(line)

    def flush(self):
        if self._buf.strip():
            self._route_line(self._buf)
            self._buf = ""
        self._stream.flush()

    def fileno(self):
        return getattr(self._stream, "fileno")()

    def isatty(self):
        method = getattr(self._stream, "isatty", None)
        return bool(method()) if method is not None else False

    def writable(self):
        method = getattr(self._stream, "writable", None)
        return bool(method()) if method is not None else True

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self._stream, "errors", None)


class MultiStreamTee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        method = getattr(self._streams[0], "isatty", None)
        return bool(method()) if method is not None else False

    def writable(self):
        method = getattr(self._streams[0], "writable", None)
        return bool(method()) if method is not None else True

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", None)

    @property
    def errors(self):
        return getattr(self._streams[0], "errors", None)


__all__ = ["LineRoutingTee", "MultiStreamTee"]
