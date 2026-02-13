#!/usr/bin/env python

import sys
from typing import Any

import tomllib

from experiments.experiment_sampler import ExperimentConfig, sampler, scan_local

_CONFIG_KEY = "config"
_REQUIRED_KEYS = (
    "exp_dir",
    "env_tag",
    "opt_name",
    "num_arms",
    "num_rounds",
    "num_reps",
)
_OPTIONAL_KEYS = (
    "num_denoise",
    "num_denoise_passive",
    "max_proposal_seconds",
    "max_total_seconds",
    "b_trace",
)
_ALL_EXPERIMENT_KEYS = set(_REQUIRED_KEYS + _OPTIONAL_KEYS)
_VALID_CLI_FLAGS = sorted({f"--{k.replace('_', '-')}" for k in _ALL_EXPERIMENT_KEYS} | {f"--{_CONFIG_KEY}"})


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


def _raise_unknown_arg(key: str) -> None:
    raise ValueError(f"Unknown argument {key}. Valid: {_VALID_CLI_FLAGS}")


def _coerce_mapping_keys(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("TOML config must be a mapping at root or under [experiment].")

    out: dict[str, Any] = {}
    for key, value in raw.items():
        norm = _normalize_key(str(key))
        if norm not in _ALL_EXPERIMENT_KEYS:
            raise ValueError(f"Unknown key '{key}' in {source}. Valid keys: {sorted(_ALL_EXPERIMENT_KEYS)}")
        out[norm] = value
    return out


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    section = data.get("experiment", data)
    return _coerce_mapping_keys(section, source=f"TOML '{path}'")


def _parse_cli_overrides(argv: list[str]) -> tuple[str | None, dict[str, Any]]:
    out: dict[str, Any] = {}
    config_path = None
    i = 0

    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            raise ValueError(f"Argument must start with '--': {token}")
        if "=" in token:
            key, value = token.split("=", 1)
        else:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {token}")
            key = token
            value = argv[i + 1]
            i += 1
        norm = _normalize_key(key[2:])
        if norm == _CONFIG_KEY:
            config_path = value
        else:
            if norm not in _ALL_EXPERIMENT_KEYS:
                _raise_unknown_arg(key)
            out[norm] = value
        i += 1

    return config_path, out


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_KEYS)}")


def load_experiment_config(argv: list[str]) -> ExperimentConfig:
    config_path, cli_cfg = _parse_cli_overrides(argv)
    toml_cfg = _load_toml_config(config_path) if config_path else {}
    merged = {**toml_cfg, **cli_cfg}
    _validate_required(merged)
    return ExperimentConfig.from_dict(merged)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    config = load_experiment_config(args)
    sampler(config, distributor_fn=scan_local)


if __name__ == "__main__":
    main()
