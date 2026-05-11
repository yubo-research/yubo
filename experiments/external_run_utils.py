from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from common.mapping_keys import coerce_mapping_keys, normalize_toml_key


def normalize_key(key: str) -> str:
    return normalize_toml_key(key)


def normalize_mapping(raw: dict[str, Any], *, source: str, valid_keys: set[str]) -> dict[str, Any]:
    return coerce_mapping_keys(raw, source=source, valid_keys=valid_keys, not_mapping_msg=f"{source} must be a mapping.")


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def parse_toml_value(raw: str) -> Any:
    from common.config_toml import parse_value

    return parse_value(raw)


def parse_section_overrides(
    override_strings: tuple[str, ...],
    *,
    valid_by_section: dict[str, set[str]],
    freeform_sections: set[str] | None = None,
    value_validator: Callable[[str, str, Any], None] | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    freeform_sections = freeform_sections or set()
    valid_sections = set(valid_by_section) | freeform_sections

    for item in override_strings:
        if "=" not in item:
            raise ValueError(f"Override must be section.key=value, got: {item}")
        key_raw, value_raw = item.split("=", 1)
        parts = [normalize_key(part.strip()) for part in key_raw.split(".")]
        if len(parts) < 2:
            raise ValueError(f"Override key must include a section, got: {key_raw}")

        section = parts[0]
        if section not in valid_sections:
            raise ValueError(f"Unknown override section '{section}'. Valid sections: {sorted(valid_sections)}")
        if section not in freeform_sections and parts[1] not in valid_by_section[section]:
            raise ValueError(f"Unknown override key '{parts[1]}' for [{section}]. Valid keys: {sorted(valid_by_section[section])}")

        value = parse_toml_value(value_raw.strip())
        if value_validator is not None:
            value_validator(section, key_raw, value)

        cur = overrides.setdefault(section, {})
        for part in parts[1:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return overrides


def abs_path(path: str | os.PathLike[str], *, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def optional_abs_path(value: Any, *, base: Path) -> str | None:
    if value in (None, ""):
        return None
    return str(abs_path(str(value), base=base))


def log_path(exp_dir: Path, value: Any, *, default: str) -> Path:
    raw = str(value or default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else exp_dir / path


def string_env_vars(raw: Any) -> dict[str, str]:
    env_vars = raw or {}
    if not isinstance(env_vars, dict):
        raise TypeError("[env].vars must be a mapping of environment variable names to values.")
    if not all(isinstance(k, str) for k in env_vars):
        raise TypeError("[env].vars keys must be strings.")
    return {str(k): str(v) for k, v in env_vars.items()}


def quote_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def append_cli_arg(cmd: list[str], key: str, value: Any, *, bool_key: bool | None = None) -> None:
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            append_cli_arg(cmd, key, item, bool_key=bool_key)
        return

    flag = "--" + key.replace("_", "-")
    is_bool_flag = isinstance(value, bool) if bool_key is None else bool_key
    if is_bool_flag:
        cmd.append(flag if bool(value) else "--no-" + key.replace("_", "-"))
    else:
        cmd.extend([flag, str(value)])


def write_metadata(exp_dir: Path, cfg: dict[str, Any], cmd: list[str]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(exp_dir / "command.json", "w", encoding="utf-8") as f:
        json.dump(cmd, f, indent=2)
    with open(exp_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(quote_command(cmd))
        f.write("\n")


def run_with_log(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()
