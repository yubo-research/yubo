#!/usr/bin/env python

from typing import TYPE_CHECKING, Any

import click
import tomllib

if TYPE_CHECKING:
    from experiments.experiment_sampler import ExperimentConfig

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
    "video_enable",
    "video_num_episodes",
    "video_num_video_episodes",
    "video_episode_selection",
    "video_seed_base",
    "video_prefix",
)
_ALL_EXPERIMENT_KEYS = set(_REQUIRED_KEYS + _OPTIONAL_KEYS)


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


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    section = data.get("experiment", data)
    return _coerce_mapping_keys(section, source=f"TOML '{path}'")


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_KEYS)}")


def load_experiment_config(*, config_toml_path: str) -> "ExperimentConfig":
    from experiments.experiment_sampler import ExperimentConfig

    cfg = _load_toml_config(config_toml_path)
    _validate_required(cfg)
    return ExperimentConfig.from_dict(cfg)


@click.group(help="Run experiments from a TOML config.")
def cli() -> None:
    pass


@cli.command(help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
def local(config_toml: str) -> None:
    from experiments.experiment_sampler import sampler, scan_local

    try:
        config = load_experiment_config(config_toml_path=config_toml)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    sampler(config, distributor_fn=scan_local)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
