from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from llm.episode_types import Case, Signal, Turn


def format_signal_log(case: Case, signal: Signal) -> str:
    lines = []
    case_id = getattr(case, "id", None)
    if case_id is not None:
        lines.append(f"CASE: {case_id}")
    lines.extend(_format_prompt_block(case.prompt))
    lines.append(f"SUMMARY: reward={float(signal.reward):.4f} status={signal.status}")
    if signal.metrics:
        lines.append(f"METRICS: {_compact_json(signal.metrics)}")
    lines.append("TRACE:")
    for turn in signal.turns:
        lines.extend(format_turn(turn))
    if signal.error:
        lines.append(f"ERROR: {signal.error}")
    return "\n".join(lines)


def format_turn(turn: Turn) -> list[str]:
    role_tag = _role_tag(turn.kind)
    display_prefix = role_tag if turn.name is None else f"{role_tag} [{turn.name}]"
    lines = [f"- {display_prefix}"]
    lines.extend(_prefixed_lines(display_prefix, turn.text))
    metadata = _format_metadata_lines(role_tag, turn.data)
    if metadata:
        lines.append("  META:")
        lines.extend([f"    {line}" for line in metadata])
    return lines


def format_step_block(turn_idx: int, step_data: dict[str, Any]) -> list[str]:
    role_tag = _role_tag(_value(step_data, "role", "unknown"))
    lines = [f"[turn {turn_idx}] {role_tag.lower()}"]
    if role_tag == "ASSISTANT":
        content = _value(step_data, "content", _value(step_data, "output", ""))
        lines.append(f"- {role_tag}")
        lines.extend(_prefixed_lines(role_tag, content))
        metadata = _format_metadata_lines(role_tag, _step_metadata(step_data))
        if metadata:
            lines.append("  META:")
            lines.extend([f"    {line}" for line in metadata])
        return lines
    if role_tag == "TOOL":
        name = str(_value(step_data, "name", "tool") or "tool")
        content = _value(step_data, "output", _value(step_data, "content", ""))
        lines.append(f"- TOOL [{name}]")
        lines.extend(_prefixed_lines(f"TOOL [{name}]", content))
        metadata = _format_metadata_lines(role_tag, _step_metadata(step_data))
        if metadata:
            lines.append("  META:")
            lines.extend([f"    {line}" for line in metadata])
        return lines
    content = _value(step_data, "content", _value(step_data, "output", ""))
    lines.append(f"- {role_tag}")
    lines.extend(_prefixed_lines(role_tag, content))
    metadata = _format_metadata_lines(role_tag, _step_metadata(step_data))
    if metadata:
        lines.append("  META:")
        lines.extend([f"    {line}" for line in metadata])
    return lines


def _format_prompt_block(prompt: Any) -> list[str]:
    if isinstance(prompt, dict) and ("role" in prompt or "content" in prompt):
        return ["PROMPT:", *_format_message(prompt)]
    if isinstance(prompt, list | tuple):
        if not prompt:
            return ["PROMPT:"]
        lines = ["PROMPT:"]
        for message in prompt:
            lines.extend(_format_message(message))
        return lines
    return _prefixed_lines("PROMPT", prompt)


def _format_message(message: Any) -> list[str]:
    if isinstance(message, dict):
        role_tag = _role_tag(message.get("role", "user"))
        display_prefix = role_tag if message.get("name") is None else f"{role_tag} [{message.get('name')}]"
        lines = [f"- {display_prefix}"]
        lines.extend(_indented_lines(message.get("content", "")))
        metadata = _format_metadata_lines(role_tag, _message_metadata(message))
        if metadata:
            lines.append("  META:")
            lines.extend([f"    {line}" for line in metadata])
        return lines
    return ["- USER", *_indented_lines(message)]


def _format_metadata_lines(role_tag: str, metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    data = dict(metadata)
    lines: list[str] = []

    reasoning_content = data.pop("reasoning_content", None)
    if reasoning_content is not None:
        reasoning_lines = _stringify_text(reasoning_content).splitlines() or [""]
        lines.append(f"{role_tag}_REASONING: {reasoning_lines[0]}")
        lines.extend([f"  {line}" for line in reasoning_lines[1:]])

    tool_calls = data.pop("tool_calls", None)
    if tool_calls is not None:
        for tool_call in tool_calls:
            lines.append(f"{role_tag}_TOOL_CALL: {_format_tool_call(tool_call)}")

    finish_reason = data.pop("finish_reason", None)
    if finish_reason is not None:
        lines.append(f"{role_tag}_FINISH: {_stringify_scalar(finish_reason)}")

    is_truncated = data.pop("is_truncated", None)
    if is_truncated is not None:
        lines.append(f"{role_tag}_TRUNCATED: {str(bool(is_truncated)).lower()}")

    tool_call_id = data.pop("tool_call_id", None)
    if tool_call_id is not None:
        lines.append(f"{role_tag}_TOOL_CALL_ID: {_stringify_scalar(tool_call_id)}")

    if data:
        lines.append(f"{role_tag}_DATA: {_compact_json(data)}")

    return lines


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("reasoning_content", "tool_calls", "finish_reason", "is_truncated", "tool_call_id"):
        value = message.get(key)
        if value is not None:
            metadata[key] = value
    extra = message.get("data")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is not None and key not in metadata:
                metadata[key] = value
    return metadata


def _step_metadata(step_data: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("reasoning_content", "tool_calls", "finish_reason", "is_truncated", "tool_call_id"):
        value = step_data.get(key)
        if value is not None:
            metadata[key] = value
    extra = step_data.get("data")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is not None and key not in metadata:
                metadata[key] = value
    for key, value in step_data.items():
        if key not in {"role", "name", "content", "output", "data", "turn_idx"} and value is not None and key not in metadata:
            metadata[key] = value
    return metadata


def _format_tool_call(tool_call: Any) -> str:
    normalized = _normalize_value(tool_call)
    if isinstance(normalized, dict):
        function = normalized.get("function")
        if isinstance(function, dict):
            name = normalized.get("name", function.get("name"))
            arguments = normalized.get("arguments", function.get("arguments"))
        else:
            name = normalized.get("name")
            arguments = normalized.get("arguments")
        summary: dict[str, Any] = {}
        if normalized.get("id") is not None:
            summary["id"] = normalized["id"]
        if name is not None:
            summary["name"] = name
        if arguments is not None:
            summary["arguments"] = arguments
        if summary:
            return _compact_json(summary)
        return _compact_json(normalized)
    return _stringify_scalar(normalized)


def _prefixed_lines(prefix: str, value: Any) -> list[str]:
    text = _stringify_text(value)
    if not text:
        return [f"{prefix}:"]
    return [f"{prefix}: {line}" for line in text.splitlines() or [""]]


def _indented_lines(value: Any) -> list[str]:
    text = _stringify_text(value)
    if not text:
        return ["  <empty>"]
    return [f"  {line}" for line in text.splitlines() or [""]]


def _stringify_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value or "content" in value:
            return _stringify_text(value.get("text", value.get("content")))
        return _compact_json(value)
    if isinstance(value, list | tuple):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_stringify_text(item.get("text", item.get("content", item))))
            else:
                parts.append(_stringify_text(getattr(item, "text", getattr(item, "content", item))))
        parts = [part for part in parts if part]
        if parts:
            return "\n".join(parts)
        return _compact_json(value)
    return _stringify_scalar(value)


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _compact_json(value: Any) -> str:
    return json.dumps(_normalize_value(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _normalize_value(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _normalize_value(model_dump(exclude_none=True))
        except TypeError:
            return _normalize_value(model_dump())
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items() if item is not None}
    if isinstance(value, list | tuple | set | frozenset):
        return [_normalize_value(item) for item in value]
    if hasattr(value, "__dict__"):
        data = {key: item for key, item in vars(value).items() if not key.startswith("_") and item is not None}
        if data:
            return _normalize_value(data)
    return str(value)


def _role_tag(kind: Any) -> str:
    role = str(kind or "unknown").strip().upper()
    return role or "UNKNOWN"


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


__all__ = [
    "format_signal_log",
    "format_step_block",
    "format_turn",
]
