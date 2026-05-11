#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import click

from experiments.external_run_utils import (
    abs_path,
    append_cli_arg,
    deep_update,
    log_path,
    normalize_key,
    normalize_mapping,
    optional_abs_path,
    parse_section_overrides,
    quote_command,
    run_with_log,
    string_env_vars,
    write_metadata,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_EXPERIMENT_KEYS = {
    "exp_dir",
    "repo_dir",
    "hf_home",
    "log_file",
    "dry_run",
}

_NANOEGG_KEYS = {
    "seed",
    "dtype",
    "batch_size",
    "population_size",
    "num_epochs",
    "alpha",
    "sigma_shift",
    "use_clt",
    "fast_fitness",
    "alpha_decay_timestep",
    "n_layer",
    "n_embd",
    "vocab_size",
    "noise_reuse",
    "tokens_per_update",
    "dir_path",
    "train_output_path",
    "valid_output_path",
    "test_output_path",
    "regenerate_model",
    "wandb_project",
    "tag",
    "track",
    "validate_every",
    "validation_batch_size",
    "coord_addr",
    "num_procs",
    "proc_id",
}

_BOOL_KEYS = {
    "use_clt",
    "fast_fitness",
    "regenerate_model",
    "track",
}

_DATA_KEYS = {
    "synthetic",
    "seed",
    "train_bytes",
    "valid_bytes",
    "test_bytes",
}

_ENV_KEYS = {"vars"}


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    valid_sections = {"experiment", "nanoegg", "data", "env"}
    unknown_sections = [key for key in raw if normalize_key(str(key)) not in valid_sections]
    if unknown_sections:
        raise ValueError(f"Unknown TOML section(s): {unknown_sections}. Valid sections: {sorted(valid_sections)}")

    if "experiment" not in raw:
        raise ValueError("Missing required [experiment] section.")
    if "nanoegg" not in raw:
        raise ValueError("Missing required [nanoegg] section.")

    cfg = {
        "experiment": normalize_mapping(raw["experiment"], source="[experiment]", valid_keys=_EXPERIMENT_KEYS),
        "nanoegg": normalize_mapping(raw["nanoegg"], source="[nanoegg]", valid_keys=_NANOEGG_KEYS),
        "data": normalize_mapping(raw.get("data", {}), source="[data]", valid_keys=_DATA_KEYS),
        "env": normalize_mapping(raw.get("env", {}), source="[env]", valid_keys=_ENV_KEYS),
    }
    return cfg


def _parse_overrides(override_strings: tuple[str, ...]) -> dict[str, Any]:
    return parse_section_overrides(
        override_strings,
        valid_by_section={
            "experiment": _EXPERIMENT_KEYS,
            "nanoegg": _NANOEGG_KEYS,
            "data": _DATA_KEYS,
            "env": _ENV_KEYS,
        },
    )


def _default_nanoegg_args(exp_dir: Path) -> dict[str, Any]:
    return {
        "seed": 0,
        "dtype": "int8",
        "batch_size": 4,
        "population_size": 1024,
        "num_epochs": 1000,
        "alpha": 4.0,
        "sigma_shift": 4,
        "use_clt": True,
        "fast_fitness": True,
        "alpha_decay_timestep": 1000,
        "n_layer": 6,
        "n_embd": 256,
        "vocab_size": 256,
        "noise_reuse": 1,
        "tokens_per_update": 100,
        "dir_path": str(exp_dir / "cached_files"),
        "train_output_path": "minipile_train.npy",
        "valid_output_path": "minipile_valid.npy",
        "test_output_path": "minipile_test.npy",
        "regenerate_model": False,
        "wandb_project": "HyperscalePretraining1",
        "tag": "",
        "track": False,
        "validate_every": 10,
        "validation_batch_size": 1024,
        "coord_addr": None,
        "num_procs": None,
        "proc_id": None,
    }


def _finalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    exp = cfg["experiment"]
    if "exp_dir" not in exp:
        raise ValueError("[experiment].exp_dir is required.")
    exp_dir = abs_path(str(exp["exp_dir"]), base=_PROJECT_ROOT)
    repo_dir = abs_path(str(exp.get("repo_dir", ".external/nano-egg")), base=_PROJECT_ROOT)
    hf_home = optional_abs_path(exp.get("hf_home"), base=_PROJECT_ROOT)

    nanoegg = _default_nanoegg_args(exp_dir)
    nanoegg.update(cfg["nanoegg"])
    nanoegg["dir_path"] = str(abs_path(str(nanoegg["dir_path"]), base=exp_dir))

    data = {
        "synthetic": False,
        "seed": 0,
        "train_bytes": None,
        "valid_bytes": None,
        "test_bytes": None,
    }
    data.update(cfg["data"])

    return {
        "experiment": {
            **exp,
            "exp_dir": str(exp_dir),
            "repo_dir": str(repo_dir),
            "hf_home": hf_home,
            "log_file": str(log_path(exp_dir, exp.get("log_file"), default="nanoegg.log")),
            "dry_run": bool(exp.get("dry_run", False)),
        },
        "nanoegg": nanoegg,
        "data": data,
        "env": {"vars": string_env_vars(cfg["env"].get("vars", {}))},
    }


def _make_command(repo_dir: Path, nanoegg: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(repo_dir / "run.py")]
    for key in _default_nanoegg_args(Path(".")).keys():
        append_cli_arg(cmd, key, nanoegg[key], bool_key=key in _BOOL_KEYS)
    return cmd


def _synthetic_sizes(nanoegg: dict[str, Any], data: dict[str, Any]) -> tuple[int, int, int]:
    batch_size = int(nanoegg["batch_size"])
    population_size = int(nanoegg["population_size"])
    tokens_per_update = int(nanoegg["tokens_per_update"])
    validation_batch_size = int(nanoegg["validation_batch_size"])
    total = max(population_size, 2 * batch_size)
    group_size = max(total // batch_size, 1)
    num_sequences = max(total // group_size, 1)
    min_train = num_sequences * (tokens_per_update + 1)
    min_valid = max(validation_batch_size * 2, validation_batch_size * (tokens_per_update + 1))
    train_bytes = int(data["train_bytes"] or max(min_train * 4, 1024))
    valid_bytes = int(data["valid_bytes"] or max(min_valid, 1024))
    test_bytes = int(data["test_bytes"] or max(min_valid, 1024))
    return train_bytes, valid_bytes, test_bytes


def _write_synthetic_data(nanoegg: dict[str, Any], data: dict[str, Any]) -> None:
    import numpy as np

    dir_path = Path(str(nanoegg["dir_path"]))
    dir_path.mkdir(parents=True, exist_ok=True)
    train_bytes, valid_bytes, test_bytes = _synthetic_sizes(nanoegg, data)
    rng = np.random.default_rng(int(data["seed"]))
    for filename, size in (
        (str(nanoegg["train_output_path"]), train_bytes),
        (str(nanoegg["valid_output_path"]), valid_bytes),
        (str(nanoegg["test_output_path"]), test_bytes),
    ):
        path = dir_path / filename
        arr = rng.integers(0, 256, size=int(size), dtype=np.uint8)
        np.save(path, arr)
        print(f"SYNTHETIC_DATA: wrote {path} bytes={int(size)}")


@click.group(help="Run nano-egg pretraining from a TOML config.")
def cli() -> None:
    pass


@cli.command(name="local", help="Run nano-egg pretraining from a TOML config.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("-o", "--opt", "overrides", multiple=True, help="Override section.key=value, e.g. -o nanoegg.num_epochs=1")
@click.option("--dry-run", is_flag=True, help="Print the command but do not execute nano-egg.")
def local(config_toml: str, overrides: tuple[str, ...], dry_run: bool) -> None:
    try:
        cfg = _load_toml_config(config_toml)
        if overrides:
            deep_update(cfg, _parse_overrides(overrides))
        cfg = _finalize_config(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    exp_dir = Path(cfg["experiment"]["exp_dir"])
    repo_dir = Path(cfg["experiment"]["repo_dir"])
    log_path = Path(cfg["experiment"]["log_file"])
    cmd = _make_command(repo_dir, cfg["nanoegg"])

    print(f"EXP_DIR: {exp_dir}")
    print(f"NANOEGG_REPO: {repo_dir}")
    print("COMMAND:", quote_command(cmd))

    if dry_run or bool(cfg["experiment"]["dry_run"]):
        print("DRY_RUN: true")
        return

    write_metadata(exp_dir, cfg, cmd)

    if bool(cfg["data"]["synthetic"]):
        _write_synthetic_data(cfg["nanoegg"], cfg["data"])

    if not (repo_dir / "run.py").is_file():
        raise click.ClickException(f"nano-egg run.py not found at {repo_dir / 'run.py'}")

    env = os.environ.copy()
    if cfg["experiment"]["hf_home"]:
        env["HF_HOME"] = str(cfg["experiment"]["hf_home"])
        env["HF_HUB_CACHE"] = str(cfg["experiment"]["hf_home"])
    env.update(cfg["env"]["vars"])

    rc = run_with_log(cmd, cwd=repo_dir, log_path=log_path, env=env)
    if rc != 0:
        raise click.ClickException(f"nano-egg exited with status {rc}. Log: {log_path}")
    print(f"DONE: log={log_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
