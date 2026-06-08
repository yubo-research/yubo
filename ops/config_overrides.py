from collections.abc import Callable
from typing import Any


def parse_override_value(raw: str) -> Any:
    from common.config_toml import parse_value

    return parse_value(raw)


def parse_overrides(
    override_strings: tuple[str, ...],
    *,
    valid_keys: set[str],
    normalize_key: Callable[[str], str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in override_strings:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key_raw, value_raw = item.split("=", 1)
        key = normalize_key(key_raw.strip())
        if key not in valid_keys:
            raise ValueError(f"Unknown override key '{key_raw}'. Valid keys: {sorted(valid_keys)}")
        out[key] = parse_override_value(value_raw.strip())
    return out
