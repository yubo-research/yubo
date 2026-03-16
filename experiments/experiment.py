#!/usr/bin/env python
"""CLI for experiments (BO or RL via routing)."""

from __future__ import annotations

import dataclasses
from typing import Any

import click
import tomllib

__all__ = ["cli", "local", "main", "load_experiment_config"]


def _n(k: str) -> str:
    return str(k).replace("-", "_")


def _section(data: dict) -> dict:
    s = data.get("experiment")
    if s is None:
        s = {k: v for k, v in data.items() if _n(str(k)) != "optimizer"}
    return {_n(str(k)): v for k, v in s.items()}


def _opt_from_optimizer(raw: dict, path: str) -> str:
    if not isinstance(raw, dict):
        raise TypeError(f"TOML '{path}' [optimizer] must be a mapping.")
    unknown = [k for k in raw if _n(str(k)) not in {"name", "params"}]
    if unknown:
        raise ValueError(f"Unknown key(s) in [optimizer]: {unknown}")
    name = raw.get("name")
    if not name or not str(name).strip():
        raise TypeError("'name' in [optimizer] must be non-empty string.")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("'params' in [optimizer] must be a mapping.")
    parts = [str(name).strip()]
    for k, v in params.items():
        nk = _n(str(k))
        if not nk:
            raise ValueError("Empty param key in [optimizer].params")
        if isinstance(v, bool):
            tv = "true" if v else "false"
        elif isinstance(v, (int, float, str)):
            tv = str(v)
            if "/" in tv:
                raise ValueError(f"Param '{k}' cannot contain '/'")
        else:
            raise TypeError(f"Param '{k}' must be int/float/str/bool")
        parts.append(f"{nk}={tv}")
    return "/".join(parts)


def _load_section(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = _section(data)
    opt = data.get("optimizer")
    if opt:
        if "opt_name" in section:
            raise ValueError(f"TOML '{path}' has both experiment.opt_name and [optimizer].name")
        section["opt_name"] = _opt_from_optimizer(opt, path)
    return section


def _bo_keys() -> set[str]:
    from experiments.experiment_sampler import ExperimentConfig

    return {f.name for f in dataclasses.fields(ExperimentConfig)}


def load_toml_config(path: str, *, for_bo: bool = True) -> dict[str, Any]:
    section = _load_section(path)
    if for_bo and is_rl(section):
        raise ValueError(f"TOML '{path}' looks like RL. Use `python -m experiments.experiment local ...`.")
    if for_bo:
        for k in section:
            if k not in _bo_keys():
                raise ValueError(f"Unknown key '{k}' in TOML '{path}'. Valid: {sorted(_bo_keys())}")
    return section


def _validate_required(cfg: dict) -> None:
    req = {"exp_dir", "env_tag", "opt_name", "num_arms", "num_reps"}
    missing = [k for k in req if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if not any(k in cfg for k in ("num_rounds", "total_timesteps")):
        raise ValueError("Missing budget: one of num_rounds, total_timesteps")


def _parse_overrides(strings: tuple[str, ...], strict: bool) -> dict[str, Any]:
    from common.config_toml import parse_value

    out = {}
    for s in strings:
        if "=" not in s:
            raise ValueError(f"Override must be key=value, got: {s}")
        k, v = s.split("=", 1)
        nk = _n(k.strip())
        if strict and nk not in _bo_keys():
            raise ValueError(f"Unknown override '{k}'. Valid: {sorted(_bo_keys())}")
        out[nk] = parse_value(v.strip())
    return out


def load_experiment_config(*, config_toml_path: str, overrides: dict[str, Any] | None = None):
    from experiments.experiment_sampler import ExperimentConfig

    cfg = load_toml_config(config_toml_path)
    if overrides:
        cfg = {**cfg, **overrides}
    _validate_required(cfg)
    return ExperimentConfig.from_dict(cfg)


def _is_rl_algo(opt_name: str) -> bool:
    try:
        from rl import builtins
        from rl.registry import get_algo

        builtins.register_all()
        get_algo(opt_name.split("/")[0] if opt_name else "")
        return True
    except (ImportError, ValueError):
        return False


def is_rl(section: dict) -> bool:
    opt = str(section.get("opt_name", "")).strip().lower()
    if opt and _is_rl_algo(opt):
        return True
    return bool(set(section) & {"backend", "device", "num_envs", "num_steps"})


def _rl_designer_opts(opt_name: str) -> dict[str, Any]:
    from common.config_toml import parse_value

    if "/" not in opt_name:
        return {}
    out = {}
    for part in opt_name.split("/")[1:]:
        if "=" not in part:
            if out:
                last = next(reversed(out))
                out[last] = f"{out[last]}/{part}"
            continue
        k, v = part.split("=", 1)
        nk = _n(k.strip())
        if nk:
            out[nk] = parse_value(v.strip())
    return out


def _run_rl(path: str, section: dict[str, Any], overrides: dict[str, Any]) -> None:
    from rl.runner import main as rl_main

    merged = {**section, **overrides}
    opt_name = str(merged.get("opt_name", "")).strip()
    base = opt_name.split("/")[0] if opt_name else ""
    ov = {**merged, "opt_name": base, **_rl_designer_opts(opt_name)}

    def _fmt(v):
        return "none" if v is None else ("true" if v is True else ("false" if v is False else str(v)))

    argv = ["--config", path] + [f"--set={k}={_fmt(v)}" for k, v in sorted(ov.items())]
    rl_main(argv)


def run_local(path: str, overrides: tuple[str, ...]) -> None:
    section = _load_section(path)
    ov = _parse_overrides(overrides, strict=not is_rl(section)) if overrides else {}

    if is_rl({**section, **ov}):
        _run_rl(path, section, ov)
        return

    cfg = {**section, **ov}
    _validate_required(cfg)
    for k in cfg:
        if k not in _bo_keys():
            raise ValueError(f"Unknown key '{k}' in config. Valid: {sorted(_bo_keys())}")

    from experiments.experiment_sampler import ExperimentConfig, sampler, scan_local

    config = ExperimentConfig.from_dict(cfg)
    sampler(config, distributor_fn=scan_local)


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
    try:
        run_local(config_toml, overrides)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e


local = _local


def _main() -> None:
    cli()


main = _main


if __name__ == "__main__":
    main()
