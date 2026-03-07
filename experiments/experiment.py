#!/usr/bin/env python

from typing import Any

import click
import tomllib

_BASE_REQUIRED_KEYS = (
    "exp_dir",
    "env_tag",
    "opt_name",
    "num_arms",
    "num_reps",
)
_BUDGET_KEYS = ("num_rounds", "total_timesteps")
_OPTIONAL_KEYS = (
    "num_denoise",
    "num_denoise_passive",
    "max_proposal_seconds",
    "max_total_seconds",
    "b_trace",
    "video_enable",
    "runtime_device",
    "local_workers",
)
_ALL_EXPERIMENT_KEYS = set(_BASE_REQUIRED_KEYS + _BUDGET_KEYS + _OPTIONAL_KEYS)
_OPTIMIZER_KEYS = {"name", "params"}


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


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


def _format_opt_value(value: Any, *, key: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        text = str(value)
        if "/" in text:
            raise ValueError(f"Optimizer param '{key}' cannot contain '/'.")
        return text
    raise TypeError(f"Optimizer param '{key}' must be int/float/str/bool, got {type(value).__name__}.")


def _compose_opt_name_from_optimizer(raw_opt: dict[str, Any], *, source: str) -> str:
    if not isinstance(raw_opt, dict):
        raise TypeError(f"'{source}' must be a mapping.")

    unknown = [k for k in raw_opt.keys() if _normalize_key(str(k)) not in _OPTIMIZER_KEYS]
    if unknown:
        raise ValueError(f"Unknown key(s) in {source}: {unknown}. Valid keys: ['name', 'params']")

    if "name" not in raw_opt:
        raise ValueError(f"Missing required key 'name' in {source}.")
    name = raw_opt["name"]
    if not isinstance(name, str) or name.strip() == "":
        raise TypeError(f"'{source}.name' must be a non-empty string.")

    params_raw = raw_opt.get("params", {})
    if not isinstance(params_raw, dict):
        raise TypeError(f"'{source}.params' must be a mapping.")

    if not params_raw:
        return name.strip()

    parts = [name.strip()]
    for key, value in params_raw.items():
        norm_key = _normalize_key(str(key))
        if norm_key == "":
            raise ValueError(f"Invalid empty optimizer param key in {source}.params.")
        value_text = _format_opt_value(value, key=norm_key)
        parts.append(f"{norm_key}={value_text}")
    return "/".join(parts)


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    section = data.get("experiment")
    if section is None:
        section = {key: value for key, value in data.items() if _normalize_key(str(key)) != "optimizer"}
    cfg = _coerce_mapping_keys(section, source=f"TOML '{path}'")

    raw_optimizer = data.get("optimizer")
    if raw_optimizer is not None:
        optimizer_opt_name = _compose_opt_name_from_optimizer(
            raw_optimizer,
            source=f"TOML '{path}' [optimizer]",
        )
        if "opt_name" in cfg:
            raise ValueError(f"TOML '{path}' defines both experiment.opt_name and [optimizer].name; use only one.")
        cfg["opt_name"] = optimizer_opt_name

    return cfg


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _BASE_REQUIRED_KEYS if k not in cfg]
    if missing:
        required = sorted(_BASE_REQUIRED_KEYS + _BUDGET_KEYS)
        raise ValueError(f"Missing required fields: {missing}. Required: {required}")
    if not any(k in cfg for k in _BUDGET_KEYS):
        raise ValueError(f"Missing required budget field: one of {list(_BUDGET_KEYS)}")


def _parse_override_value(raw: str) -> Any:
    from common.config_toml import parse_value

    return parse_value(raw)


def _parse_overrides(override_strings: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in override_strings:
        if "=" not in s:
            raise ValueError(f"Override must be key=value, got: {s}")
        key_raw, value_raw = s.split("=", 1)
        norm = _normalize_key(key_raw.strip())
        if norm not in _ALL_EXPERIMENT_KEYS:
            raise ValueError(f"Unknown override key '{key_raw}'. Valid keys: {sorted(_ALL_EXPERIMENT_KEYS)}")
        out[norm] = _parse_override_value(value_raw.strip())
    return out


def load_experiment_config(*, config_toml_path: str, overrides: dict[str, Any] | None = None):
    from experiments.experiment_sampler import ExperimentConfig

    cfg = _load_toml_config(config_toml_path)
    if overrides:
        cfg = {**cfg, **overrides}
    _validate_required(cfg)
    return ExperimentConfig.from_dict(cfg)


@click.group(help="Run experiments from a TOML config.")
def _cli() -> None:
    pass


cli = _cli


@_cli.command(name="local", help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "-o",
    "--opt",
    "overrides",
    multiple=True,
    help="Override config key: --opt key=value (e.g. --opt opt_name=turbo-enn-fit-ucb)",
)
def _local(config_toml: str, overrides: tuple[str, ...]) -> None:
    from experiments.experiment_sampler import sampler, scan_local

    override_dict: dict[str, Any] = {}
    if overrides:
        try:
            override_dict = _parse_overrides(overrides)
        except (TypeError, ValueError) as e:
            raise click.ClickException(str(e)) from e

    try:
        config = load_experiment_config(config_toml_path=config_toml, overrides=override_dict or None)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    # Load Atari/DM-Control support when needed (avoids pulling into experiments modal chain).
    if config.env_tag.startswith(("atari:", "dm:", "dm_control/")):
        __import__("problems.env_conf_atari_dm")

    sampler(config, distributor_fn=scan_local)


local = _local


def _main() -> None:
    cli()


main = _main


if __name__ == "__main__":
    main()
