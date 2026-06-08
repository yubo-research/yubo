from __future__ import annotations

import dataclasses


def dataclass_field_names(config_cls: type) -> set[str]:
    return {field.name for field in dataclasses.fields(config_cls)}


def parse_dataclass_section(raw: dict, key: str, config_cls: type, *, label: str):
    section = raw.get(key, {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError(f"{label} config field '{key}' must be a table.")
    unknown = sorted(set(section) - dataclass_field_names(config_cls))
    if unknown:
        raise ValueError(f"Unknown {label} config fields in '{key}': {', '.join(unknown)}.")
    return config_cls(**section)
