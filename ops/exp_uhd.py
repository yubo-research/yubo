#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

import click
import tomllib

_REQUIRED_TOML_KEYS = ("env_tag", "num_rounds")
_OPTIONAL_TOML_KEYS = ("lr", "perturb", "log_interval", "accuracy_interval", "target_accuracy")
_ALL_TOML_KEYS = set(_REQUIRED_TOML_KEYS + _OPTIONAL_TOML_KEYS)


def _normalize_key(key: str) -> str:
    # Match experiments/experiment.py convention: allow hyphenated keys in TOML.
    return key.replace("-", "_")


def _coerce_mapping_keys(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("TOML config must be a mapping at root or under [uhd].")

    out: dict[str, Any] = {}
    for key, value in raw.items():
        norm = _normalize_key(str(key))
        if norm not in _ALL_TOML_KEYS:
            raise ValueError(f"Unknown key '{key}' in {source}. Valid keys: {sorted(_ALL_TOML_KEYS)}")
        out[norm] = value
    return out


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("uhd", data)
    return _coerce_mapping_keys(section, source=f"TOML '{path}'")


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_TOML_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_TOML_KEYS)}")


@click.group()
def cli():
    pass


def _parse_perturb(perturb: str) -> tuple[float | None, float | None]:
    """Parse --perturb flag into (num_dim_target, num_module_target)."""
    if perturb == "dense":
        return None, None
    if perturb.startswith("dim:"):
        return float(perturb[4:]), None
    if perturb.startswith("mod:"):
        return None, float(perturb[4:])
    msg = f"Invalid --perturb value: {perturb!r}. Use 'dense', 'dim:<n>', or 'mod:<n>'."
    raise click.BadParameter(msg)


@dataclass(frozen=True)
class UHDConfig:
    env_tag: str
    num_rounds: int
    lr: float
    num_dim_target: float | None
    num_module_target: float | None
    log_interval: int
    accuracy_interval: int
    target_accuracy: float | None


def _parse_cfg(cfg: dict[str, Any]) -> UHDConfig:
    lr_default = 0.001
    perturb_default = "dim:0.5"
    log_interval_default = 1
    accuracy_interval_default = 1000
    target_accuracy_default = None

    env_tag = str(cfg["env_tag"])
    num_rounds = int(cfg["num_rounds"])
    lr = float(cfg.get("lr", lr_default))
    perturb = str(cfg.get("perturb", perturb_default))
    log_interval = int(cfg.get("log_interval", log_interval_default))
    accuracy_interval = int(cfg.get("accuracy_interval", accuracy_interval_default))
    target_accuracy = cfg.get("target_accuracy", target_accuracy_default)
    if target_accuracy is not None:
        target_accuracy = float(target_accuracy)
    ndt, nmt = _parse_perturb(perturb)
    return UHDConfig(
        env_tag=env_tag,
        num_rounds=num_rounds,
        lr=lr,
        num_dim_target=ndt,
        num_module_target=nmt,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
    )


@cli.command(help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
def local(config_toml: str) -> None:
    from ops.uhd_setup import make_loop

    try:
        cfg = _load_toml_config(config_toml)
        _validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = _parse_cfg(cfg)
    loop = make_loop(
        parsed.env_tag,
        parsed.num_rounds,
        lr=parsed.lr,
        sigma=0.001,
        num_dim_target=parsed.num_dim_target,
        num_module_target=parsed.num_module_target,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
    )
    loop.run()


@cli.command(name="modal", help="Run on Modal. Streams to stdout; optionally saves to --log-file.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--log-file", type=click.Path(dir_okay=False), default=None, help="Also save log to this local file.")
@click.option("--gpu", type=str, default="A100", help="Modal GPU type (e.g. T4, A10, A100, H100).")
def modal_cmd(config_toml: str, log_file: str | None, gpu: str) -> None:
    from ops.modal_uhd import run as modal_run

    try:
        cfg = _load_toml_config(config_toml)
        _validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = _parse_cfg(cfg)
    log_text = modal_run(
        parsed.env_tag,
        parsed.num_rounds,
        parsed.lr,
        parsed.num_dim_target,
        parsed.num_module_target,
        gpu=gpu,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
    )

    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(log_text)
        click.echo(f"Log saved to {log_file}")


if __name__ == "__main__":
    cli()
