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

_SCRIPT_MODULES = {
    "general_do_evolution": "llm_experiments.general_do_evolution",
    "general_do_evolution_multi_gpu": "llm_experiments.general_do_evolution_multi_gpu",
    "sft_evolution": "llm_experiments.sft_evolution",
    "do_grpo": "llm_experiments.do_grpo",
    "do_grpo_multi_gpu": "llm_experiments.do_grpo_multi_gpu",
}

_EXPERIMENT_KEYS = {
    "exp_dir",
    "repo_dir",
    "hf_home",
    "log_file",
    "dry_run",
}

_HYPERSCALEES_KEYS = {
    "script",
    "profile",
    "raw_args",
}

_ENV_KEYS = {"vars"}


def _supported_tasks() -> tuple[str, ...]:
    from problems.pre_obj import supported_hyperscalees_llm_bandit_tasks

    return supported_hyperscalees_llm_bandit_tasks()


def _template_config(
    *,
    task: str,
    model_choice: str,
    script: str,
    noiser: str,
    exp_dir: str,
    num_epochs: int,
    parallel_generations_per_gpu: int,
    generations_per_prompt: int,
    thinking_length: int,
    answer_length: int,
    track: bool,
    repo_dir: str,
) -> str:
    if script not in _SCRIPT_MODULES:
        raise ValueError(f"Unknown HyperscaleES script '{script}'. Valid scripts: {sorted(_SCRIPT_MODULES)}")
    tasks = set(_supported_tasks())
    if task not in tasks:
        raise ValueError(f"Unsupported HyperscaleES LLM bandit task '{task}'. Known tasks: {', '.join(sorted(tasks))}.")
    wandb_name = f"{task}_{model_choice.replace('.', 'p').lower()}_{noiser}"
    lines = [
        "# Run:",
        "# ./ops/hyperscalees_llm.py local <this-file>",
        "",
        "[experiment]",
        f'exp_dir = "{exp_dir}"',
        f'repo_dir = "{repo_dir}"',
        'log_file = "hyperscalees.log"',
        "dry_run = false",
        "",
        "[hyperscalees]",
        f'script = "{script}"',
        "",
        "[args]",
        "seed = 0",
        f'model_choice = "{model_choice}"',
        f'task = "{task}"',
        f'noiser = "{noiser}"',
        f"num_epochs = {int(num_epochs)}",
        f"parallel_generations_per_gpu = {int(parallel_generations_per_gpu)}",
        f"generations_per_prompt = {int(generations_per_prompt)}",
        f"thinking_length = {int(thinking_length)}",
        f"answer_length = {int(answer_length)}",
        "validate_every = 10",
        "parallel_validations = 8",
        "validation_iterations = 1",
        "log_output_every = 1",
        f'output_directory = "{exp_dir}/outputs"',
        f'wandb_directory = "{exp_dir}"',
        'wandb_mode = "offline"',
        f'wandb_name = "{wandb_name}"',
        f"track = {str(bool(track)).lower()}",
        "",
        "[env.vars]",
        'XLA_PYTHON_CLIENT_PREALLOCATE = "false"',
        "",
    ]
    return "\n".join(lines)


def _normalize_args(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("[args] must be a mapping.")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        norm = normalize_key(str(key))
        _validate_arg_value(value, source=f"[args].{key}")
        out[norm] = value
    return out


def _validate_arg_value(value: Any, *, source: str) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if item is None or isinstance(item, bool | int | float | str):
                continue
            raise TypeError(f"{source}[{idx}] has unsupported type {type(item).__name__}.")
        return
    raise TypeError(f"{source} has unsupported type {type(value).__name__}.")


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    valid_sections = {"experiment", "hyperscalees", "args", "env"}
    unknown_sections = [key for key in raw if normalize_key(str(key)) not in valid_sections]
    if unknown_sections:
        raise ValueError(f"Unknown TOML section(s): {unknown_sections}. Valid sections: {sorted(valid_sections)}")
    if "experiment" not in raw:
        raise ValueError("Missing required [experiment] section.")
    if "hyperscalees" not in raw:
        raise ValueError("Missing required [hyperscalees] section.")

    cfg = {
        "experiment": normalize_mapping(raw["experiment"], source="[experiment]", valid_keys=_EXPERIMENT_KEYS),
        "hyperscalees": normalize_mapping(raw["hyperscalees"], source="[hyperscalees]", valid_keys=_HYPERSCALEES_KEYS),
        "args": _normalize_args(raw.get("args", {})),
        "env": normalize_mapping(raw.get("env", {}), source="[env]", valid_keys=_ENV_KEYS),
    }
    return cfg


def _parse_overrides(override_strings: tuple[str, ...]) -> dict[str, Any]:
    def validate_value(section: str, key_raw: str, value: Any) -> None:
        if section == "args":
            _validate_arg_value(value, source=f"override {key_raw}")

    return parse_section_overrides(
        override_strings,
        valid_by_section={
            "experiment": _EXPERIMENT_KEYS,
            "hyperscalees": _HYPERSCALEES_KEYS,
            "env": _ENV_KEYS,
        },
        freeform_sections={"args"},
        value_validator=validate_value,
    )


def _finalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    exp = cfg["experiment"]
    if "exp_dir" not in exp:
        raise ValueError("[experiment].exp_dir is required.")
    exp_dir = abs_path(str(exp["exp_dir"]), base=_PROJECT_ROOT)
    repo_dir = abs_path(str(exp.get("repo_dir", ".external/HyperscaleES")), base=_PROJECT_ROOT)
    hf_home = optional_abs_path(exp.get("hf_home"), base=_PROJECT_ROOT)

    hyper = dict(cfg["hyperscalees"])
    script = str(hyper.get("script", "general_do_evolution"))
    if script not in _SCRIPT_MODULES:
        raise ValueError(f"Unknown HyperscaleES script '{script}'. Valid scripts: {sorted(_SCRIPT_MODULES)}")
    raw_args = hyper.get("raw_args", [])
    if raw_args is None:
        raw_args = []
    if not isinstance(raw_args, list) or not all(isinstance(x, str) for x in raw_args):
        raise TypeError("[hyperscalees].raw_args must be a list of strings.")
    hyper["raw_args"] = raw_args

    args = dict(cfg["args"])
    for key in ("output_directory", "wandb_directory"):
        value = args.get(key)
        if isinstance(value, str) and value.strip() != "":
            args[key] = str(abs_path(value, base=_PROJECT_ROOT))

    return {
        "experiment": {
            **exp,
            "exp_dir": str(exp_dir),
            "repo_dir": str(repo_dir),
            "hf_home": hf_home,
            "log_file": str(log_path(exp_dir, exp.get("log_file"), default="hyperscalees.log")),
            "dry_run": bool(exp.get("dry_run", False)),
        },
        "hyperscalees": hyper,
        "args": args,
        "env": {"vars": string_env_vars(cfg["env"].get("vars", {}))},
    }


def _make_command(hyper: dict[str, Any], args: dict[str, Any]) -> list[str]:
    module = _SCRIPT_MODULES[str(hyper["script"])]
    cmd = [sys.executable, "-m", module]
    for key in sorted(args):
        append_cli_arg(cmd, key, args[key])
    cmd.extend(hyper.get("raw_args", []))
    return cmd


@click.group(help="Run upstream HyperscaleES LLM experiments from a TOML config.")
def cli() -> None:
    pass


@cli.command(name="scripts", help="List supported upstream HyperscaleES LLM script entrypoints.")
def scripts() -> None:
    for name in sorted(_SCRIPT_MODULES):
        print(name)


@cli.command(name="tasks", help="List supported upstream HyperscaleES LLM bandit tasks.")
def tasks() -> None:
    for task in _supported_tasks():
        print(task)


@cli.command(name="template", help="Print a runnable TOML template for any supported HyperscaleES LLM bandit task.")
@click.option("--task", required=True, help="HyperscaleES LLM bandit task, e.g. basic_arithmetic or zebra_puzzles.")
@click.option("--model-choice", default="7w1.5B", show_default=True, help="Upstream HyperscaleES model_choice value.")
@click.option("--script", default="general_do_evolution", show_default=True, help="Upstream script name.")
@click.option("--noiser", default="eggroll", show_default=True, help="Upstream HyperscaleES noiser.")
@click.option("--exp-dir", default=None, help="Run directory to write into.")
@click.option("--repo-dir", default=".external/HyperscaleES", show_default=True, help="HyperscaleES repo path.")
@click.option("--num-epochs", default=1000, show_default=True, type=int)
@click.option("--parallel-generations-per-gpu", default=32, show_default=True, type=int)
@click.option("--generations-per-prompt", default=2, show_default=True, type=int)
@click.option("--thinking-length", default=100, show_default=True, type=int)
@click.option("--answer-length", default=100, show_default=True, type=int)
@click.option("--track/--no-track", default=False, show_default=True)
def template(
    task: str,
    model_choice: str,
    script: str,
    noiser: str,
    exp_dir: str | None,
    repo_dir: str,
    num_epochs: int,
    parallel_generations_per_gpu: int,
    generations_per_prompt: int,
    thinking_length: int,
    answer_length: int,
    track: bool,
) -> None:
    if exp_dir is None:
        model_tag = model_choice.replace(".", "p").lower()
        exp_dir = f"runs/yubo_hyperscalees/llm_bandits/{task}_{model_tag}_{noiser}"
    try:
        print(
            _template_config(
                task=task,
                model_choice=model_choice,
                script=script,
                noiser=noiser,
                exp_dir=exp_dir,
                num_epochs=num_epochs,
                parallel_generations_per_gpu=parallel_generations_per_gpu,
                generations_per_prompt=generations_per_prompt,
                thinking_length=thinking_length,
                answer_length=answer_length,
                track=track,
                repo_dir=repo_dir,
            ),
            end="",
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command(name="local", help="Run an upstream HyperscaleES LLM experiment from TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("-o", "--opt", "overrides", multiple=True, help="Override section.key=value, e.g. -o args.num_epochs=1")
@click.option("--dry-run", is_flag=True, help="Print the command but do not execute HyperscaleES.")
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
    cmd = _make_command(cfg["hyperscalees"], cfg["args"])

    print(f"EXP_DIR: {exp_dir}")
    print(f"HYPERSCALEES_REPO: {repo_dir}")
    print(f"SCRIPT: {cfg['hyperscalees']['script']}")
    print("COMMAND:", quote_command(cmd))

    env = os.environ.copy()
    if cfg["experiment"]["hf_home"]:
        env["HF_HOME"] = str(cfg["experiment"]["hf_home"])
        env["HF_HUB_CACHE"] = str(cfg["experiment"]["hf_home"])
    profile = cfg["hyperscalees"].get("profile")
    if profile not in (None, ""):
        env["PROFILE"] = str(profile)
    env.update(cfg["env"]["vars"])

    if dry_run or cfg["experiment"]["dry_run"]:
        print("DRY_RUN: true")
        return

    write_metadata(exp_dir, cfg, cmd)

    if not repo_dir.exists():
        raise click.ClickException(f"HyperscaleES repo not found: {repo_dir}")

    return_code = run_with_log(cmd, cwd=repo_dir, log_path=log_path, env=env)
    if return_code != 0:
        raise click.ClickException(f"HyperscaleES command failed with exit code {return_code}. Log: {log_path}")
    print(f"LOG: {log_path}")


if __name__ == "__main__":
    cli()
