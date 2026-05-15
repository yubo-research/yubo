from __future__ import annotations

import re


def channel_for_step(step_data: dict[str, object]) -> str:
    role = str(step_data.get("role", ""))
    return "inference" if role in {"assistant", "tool"} else "train"


def classify_console_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "diagnostics"
    if stripped.startswith(("PROMPT:", "REWARD:", "TRAJECTORY:", "Model:", "Tool [", ">>>")):
        return "inference"
    if stripped.startswith(_TRAIN_PREFIXES):
        return "train"
    return "diagnostics"


def clean_text(text: str) -> str:
    text = _ANSI_RE.sub("", text).replace("\r", "")
    return "".join(ch if ch == "\n" or ch == "\t" or ord(ch) >= 32 else " " for ch in text)


def is_attention_diagnostic(line: str) -> bool:
    lower = line.lower()
    return "error" in lower or "warning" in lower or "traceback" in lower or "exception" in lower


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_TRAIN_PREFIXES = (
    "EVAL:",
    "ITER:",
    "EXP_DIR:",
    "LOG:",
    "COMMAND:",
    "RESULT:",
    "LLM:",
    "LLM_EGGROLL:",
    "UNIVERSAL:",
    "DRY_RUN:",
    "HYPERSCALEES_REPO:",
    "SCRIPT:",
    "Running ",
    "REP ",
    "Epsilon",
    "BSZO:",
    "UHD-",
)


__all__ = ["channel_for_step", "classify_console_line", "clean_text", "is_attention_diagnostic"]
