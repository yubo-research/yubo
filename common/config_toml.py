import os
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib


def load_toml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return cfg


def parse_value(raw: str) -> Any:
    if raw.lower() in {"none", "null"}:
        return None
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_set_args(argv: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--set":
            if i + 1 >= len(argv):
                raise ValueError("Expected KEY=VALUE after --set")
            kv = argv[i + 1]
            if "=" not in kv:
                raise ValueError(f"Invalid --set value: {kv}")
            k, v = kv.split("=", 1)
            overrides[k] = parse_value(v)
            i += 2
            continue
        if arg.startswith("--set="):
            kv = arg[len("--set=") :]
            if "=" not in kv:
                raise ValueError(f"Invalid --set value: {kv}")
            k, v = kv.split("=", 1)
            overrides[k] = parse_value(v)
            i += 1
            continue
        raise ValueError(f"Unknown argument: {arg}")
    return overrides
