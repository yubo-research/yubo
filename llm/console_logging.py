from __future__ import annotations

import logging
from typing import Any

from llm.console_text import classify_console_line


class ConsoleLoggingContext:
    def __init__(self, observer: Any) -> None:
        self.observer = observer
        self.handler = ConsoleLogHandler(observer)
        self.old_levels: dict[str, int] = {}

    def __enter__(self):
        self.handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(self.handler)
        for name in _NOISY_LOGGERS:
            logger = logging.getLogger(name)
            self.old_levels[name] = logger.level
            logger.setLevel(logging.WARNING)
        logging.captureWarnings(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        logging.getLogger().removeHandler(self.handler)
        for name, level in self.old_levels.items():
            logging.getLogger(name).setLevel(level)
        logging.captureWarnings(False)


class ConsoleLogHandler(logging.Handler):
    def __init__(self, observer: Any) -> None:
        super().__init__(level=logging.INFO)
        self.observer = observer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            channel = classify_console_line(record.getMessage())
            payload = record_payload(record)
            if channel == "inference":
                self.observer.append_inference(line, kind="log", payload=payload)
            elif channel == "train":
                self.observer.append_train(line, kind="log", payload=payload)
            else:
                self.observer.append_diagnostics(line, kind="log", payload=payload)
        except Exception:
            self.handleError(record)


def record_payload(record: logging.LogRecord) -> dict[str, Any]:
    return {
        "logger": record.name,
        "level": record.levelname,
        "message": record.getMessage(),
    }


_NOISY_LOGGERS = (
    "asyncio",
    "datasets",
    "filelock",
    "huggingface_hub",
    "numba",
    "ray",
    "torch",
    "transformers",
    "urllib3",
    "vllm",
)


__all__ = ["ConsoleLoggingContext", "ConsoleLogHandler", "record_payload"]
