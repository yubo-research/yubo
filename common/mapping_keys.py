from __future__ import annotations

from typing import Any


def normalize_toml_key(key: str) -> str:
    return str(key).replace("-", "_")


def coerce_mapping_keys(
    raw: dict[str, Any],
    *,
    source: str,
    valid_keys: set[str],
    not_mapping_msg: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(not_mapping_msg)

    out: dict[str, Any] = {}
    for key, value in raw.items():
        norm = normalize_toml_key(str(key))
        if norm not in valid_keys:
            raise ValueError(f"Unknown key '{key}' in {source}. Valid keys: {sorted(valid_keys)}")
        out[norm] = value
    return out
