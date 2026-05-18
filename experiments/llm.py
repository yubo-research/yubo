#!/usr/bin/env python
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import click
import tomllib

from common.config_toml import parse_value
from experiments.external_run_utils import abs_path, log_path
from llm.config import LLMConfig
from llm.console_observer import SplitConsoleObserver, tee_stdout_to_exp
from llm.registry import (
    resolve_llm_env,
    resolve_llm_policy,
    supported_llm_env_tags,
    supported_llm_policy_tags,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_TOML_KEYS = ("env_tag", "policy_tag", "optimizer")
_OPTIONAL_TOML_KEYS = (
    "exp_dir",
    "log_file",
    "dry_run",
    "hf_home",
    "num_rounds",
    "total_timesteps",
    "num_epochs",
    "num_reps",
    "problem_seed",
    "noise_seed_0",
    "seed_offset",
    "lr",
    "sigma",
    "log_interval",
    "target_accuracy",
    "batch_size",
    "population_size",
    "max_tokens",
    "temperature",
    "samples_per_prompt",
    "prompt_batch_size",
    "pass_at_k",
    "normalize_with_std",
    "scale_lr_in_grad",
    "steps_per_adapter",
    "num_gpus",
    "num_engines",
    "tensor_parallel_size",
    "steps_per_eval",
    "eval_batch_size",
    "use_wandb",
    "wandb_project",
    "wandb_name",
    "save_freq",
    "checkpoint_dir",
    "resume_from",
    "sub_dataset_size",
    "kl_beta",
    "reference_policy_tag",
    "pretrain_lora_only",
    "pretrain_search_dim",
    "vllm_enforce_eager",
    "vllm_max_model_len",
    "vllm_gpu_memory_utilization",
    "vllm_max_num_seqs",
    "vllm_max_num_batched_tokens",
    "vllm_speculative_method",
    "vllm_speculative_model",
    "vllm_num_speculative_tokens",
)
_ALL_TOML_KEYS = set(_REQUIRED_TOML_KEYS + _OPTIONAL_TOML_KEYS)
_DIRECT_LLM_OPTIMIZERS = {"eggroll", "sft", "rkl"}


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("llm", data)
    return _coerce_mapping_keys(section, source=f"TOML '{path}'")


def _coerce_mapping_keys(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("TOML config must be a mapping at root or under [llm].")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        norm = _normalize_key(str(key))
        if norm not in _ALL_TOML_KEYS:
            raise ValueError(f"Unknown key '{key}' in {source}. Valid keys: {sorted(_ALL_TOML_KEYS)}")
        out[norm] = value
    return out


def _parse_overrides(override_strings: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in override_strings:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key_raw, value_raw = item.split("=", 1)
        norm = _normalize_key(key_raw.strip())
        if norm not in _ALL_TOML_KEYS:
            raise ValueError(f"Unknown override key '{key_raw}'. Valid keys: {sorted(_ALL_TOML_KEYS)}")
        out[norm] = parse_value(value_raw.strip())
    return out


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_TOML_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_TOML_KEYS)}")


def _parse_cfg(cfg: dict[str, Any]) -> LLMConfig:
    _validate_required(cfg)
    env_tag = str(cfg["env_tag"])
    policy_tag = str(cfg["policy_tag"])
    optimizer = str(cfg["optimizer"])
    if optimizer == "uhd":
        raise ValueError("Use [uhd] with ./ops/exp_uhd.py for UHD text runs.")
    if optimizer not in _DIRECT_LLM_OPTIMIZERS:
        raise ValueError(f"Unknown direct LLM optimizer '{optimizer}'. Valid: {sorted(_DIRECT_LLM_OPTIMIZERS)}")

    env = resolve_llm_env(env_tag)
    policy = resolve_llm_policy(policy_tag)
    num_rounds = _optional_int(cfg, "num_rounds")
    total_timesteps = _optional_int(cfg, "total_timesteps")
    num_epochs = _optional_int(cfg, "num_epochs")
    if num_rounds is not None and num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1 (got: {num_rounds})")
    if total_timesteps is not None and total_timesteps < 1:
        raise ValueError(f"total_timesteps must be >= 1 (got: {total_timesteps})")
    if num_epochs is not None and num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1 (got: {num_epochs})")
    _validate_optimizer_budget(
        optimizer,
        num_rounds=num_rounds,
        total_timesteps=total_timesteps,
        num_epochs=num_epochs,
    )

    exp_dir = abs_path(
        str(cfg.get("exp_dir", _default_exp_dir(env_tag, policy_tag, optimizer))),
        base=_PROJECT_ROOT,
    )
    hf_home = cfg.get("hf_home")
    hf_home_resolved = None if hf_home in (None, "") else str(abs_path(str(hf_home), base=_PROJECT_ROOT))
    samples_per_prompt = int(cfg.get("samples_per_prompt", 1))
    pass_at_k = bool(cfg.get("pass_at_k", False))
    if pass_at_k and samples_per_prompt <= 1:
        raise ValueError("pass_at_k=true requires samples_per_prompt > 1.")

    tensor_parallel_size = _optional_int(cfg, "tensor_parallel_size")
    vllm_options = _parse_vllm_options(cfg)
    return LLMConfig(
        env_tag=env_tag,
        policy_tag=policy_tag,
        optimizer=optimizer,
        num_rounds=num_rounds,
        total_timesteps=total_timesteps,
        num_epochs=num_epochs,
        num_reps=int(cfg.get("num_reps", 1)),
        exp_dir=str(exp_dir),
        log_file=str(log_path(exp_dir, cfg.get("log_file"), default="llm.log")),
        dry_run=bool(cfg.get("dry_run", False)),
        hf_home=hf_home_resolved,
        problem_seed=_optional_int(cfg, "problem_seed"),
        noise_seed_0=_optional_int(cfg, "noise_seed_0"),
        seed_offset=int(cfg.get("seed_offset", 0)),
        lr=float(cfg.get("lr", 0.001)),
        sigma=float(cfg.get("sigma", 0.001)),
        log_interval=int(cfg.get("log_interval", 1)),
        target_accuracy=_optional_float(cfg, "target_accuracy"),
        batch_size=int(cfg.get("batch_size", 1)),
        population_size=int(cfg.get("population_size", 128)),
        max_tokens=int(cfg.get("max_tokens", 1024)),
        temperature=float(cfg.get("temperature", 0.0)),
        samples_per_prompt=samples_per_prompt,
        prompt_batch_size=int(cfg.get("prompt_batch_size", 2)),
        pass_at_k=pass_at_k,
        normalize_with_std=bool(cfg.get("normalize_with_std", False)),
        scale_lr_in_grad=bool(cfg.get("scale_lr_in_grad", False)),
        steps_per_adapter=int(cfg.get("steps_per_adapter", 4)),
        num_gpus=_optional_int(cfg, "num_gpus"),
        num_engines=_optional_int(cfg, "num_engines"),
        tensor_parallel_size=tensor_parallel_size,
        steps_per_eval=int(cfg.get("steps_per_eval", 10)),
        eval_batch_size=int(cfg.get("eval_batch_size", 128)),
        use_wandb=bool(cfg.get("use_wandb", False)),
        wandb_project=str(cfg.get("wandb_project", "yubo-llm")),
        wandb_name=_optional_str(cfg, "wandb_name"),
        save_freq=_optional_int(cfg, "save_freq"),
        checkpoint_dir=_optional_path(cfg, "checkpoint_dir", base=exp_dir),
        resume_from=_optional_path(cfg, "resume_from", base=_PROJECT_ROOT),
        sub_dataset_size=_optional_int(cfg, "sub_dataset_size"),
        kl_beta=_optional_float(cfg, "kl_beta"),
        reference_policy_tag=_optional_str(cfg, "reference_policy_tag"),
        pretrain_lora_only=bool(cfg.get("pretrain_lora_only", True)),
        pretrain_search_dim=int(cfg.get("pretrain_search_dim", 4096)),
        env=env,
        policy=policy,
        **vllm_options,
    )


def _default_exp_dir(env_tag: str, policy_tag: str, optimizer: str) -> str:
    return f"runs/llm/{_slug(env_tag)}_{_slug(policy_tag)}_{_slug(optimizer)}"


def _validate_optimizer_budget(
    optimizer: str,
    *,
    num_rounds: int | None,
    total_timesteps: int | None,
    num_epochs: int | None,
) -> None:
    if optimizer == "eggroll" and num_rounds is None and total_timesteps is None:
        raise ValueError("optimizer='eggroll' requires one budget field: num_rounds or total_timesteps.")
    if optimizer in {"sft", "rkl"} and num_epochs is None:
        raise ValueError(f"optimizer='{optimizer}' requires num_epochs.")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()


def _optional_int(cfg: dict[str, Any], key: str) -> int | None:
    value = cfg.get(key)
    return None if value in (None, "") else int(value)


def _optional_float(cfg: dict[str, Any], key: str) -> float | None:
    value = cfg.get(key)
    return None if value in (None, "") else float(value)


def _optional_str(cfg: dict[str, Any], key: str) -> str | None:
    value = cfg.get(key)
    return None if value in (None, "") else str(value)


def _optional_path(cfg: dict[str, Any], key: str, *, base: Path) -> str | None:
    value = cfg.get(key)
    return None if value in (None, "") else str(abs_path(str(value), base=base))


def _parse_vllm_options(cfg: dict[str, Any]) -> dict[str, int | float | str | None]:
    return {
        "vllm_enforce_eager": bool(cfg.get("vllm_enforce_eager", False)),
        "vllm_max_model_len": _optional_int(cfg, "vllm_max_model_len"),
        "vllm_gpu_memory_utilization": _optional_float(cfg, "vllm_gpu_memory_utilization"),
        "vllm_max_num_seqs": _optional_int(cfg, "vllm_max_num_seqs"),
        "vllm_max_num_batched_tokens": _optional_int(cfg, "vllm_max_num_batched_tokens"),
        "vllm_speculative_method": _optional_str(cfg, "vllm_speculative_method"),
        "vllm_speculative_model": _optional_str(cfg, "vllm_speculative_model"),
        "vllm_num_speculative_tokens": _optional_int(cfg, "vllm_num_speculative_tokens"),
    }


def _cfg_summary(cfg: LLMConfig) -> dict[str, Any]:
    tensor_parallel_size = cfg.tensor_parallel_size or cfg.policy.tensor_parallel_size
    return {
        "env_tag": cfg.env_tag,
        "policy_tag": cfg.policy_tag,
        "optimizer": cfg.optimizer,
        "task_name": cfg.env.task_name,
        "model_name": cfg.policy.model_name,
        "lora_rank": cfg.policy.lora_rank,
        "lora_alpha": cfg.policy.lora_alpha,
        "tensor_parallel_size": tensor_parallel_size,
        "num_rounds": cfg.num_rounds,
        "total_timesteps": cfg.total_timesteps,
        "num_epochs": cfg.num_epochs,
        "population_size": cfg.population_size,
        "prompt_batch_size": cfg.prompt_batch_size,
        "samples_per_prompt": cfg.samples_per_prompt,
        "pass_at_k": cfg.pass_at_k,
        "vllm_speculative_method": cfg.vllm_speculative_method,
        "vllm_speculative_model": cfg.vllm_speculative_model,
        "vllm_num_speculative_tokens": cfg.vllm_num_speculative_tokens,
        "vllm_enforce_eager": cfg.vllm_enforce_eager,
    }


@click.group(help="Run Yubo-owned LLM experiments.")
def cli() -> None:
    pass


@cli.command(name="envs", help="List supported LLM env tags.")
def envs() -> None:
    for tag in supported_llm_env_tags():
        print(tag)


@cli.command(name="policies", help="List supported LLM policy tags.")
def policies() -> None:
    for tag in supported_llm_policy_tags():
        print(tag)


@cli.command(name="template", help="Print a strict [llm] config template.")
@click.option("--env-tag", default="llm:math:gsm8k", show_default=True)
@click.option("--policy-tag", default="qwen3-1p7b-lora-r1", show_default=True)
@click.option("--optimizer", default="eggroll", show_default=True)
@click.option("--num-rounds", default=1, show_default=True, type=int)
def template(env_tag: str, policy_tag: str, optimizer: str, num_rounds: int) -> None:
    try:
        cfg = _parse_cfg(
            {
                "env_tag": env_tag,
                "policy_tag": policy_tag,
                "optimizer": optimizer,
                "num_rounds": int(num_rounds),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    print(
        "\n".join(
            [
                "# Run:",
                "# ./ops/llm.py local <this-file> --dry-run",
                "",
                "[llm]",
                f'exp_dir = "{_default_exp_dir(cfg.env_tag, cfg.policy_tag, cfg.optimizer)}"',
                'log_file = "llm.log"',
                "dry_run = false",
                f'env_tag = "{cfg.env_tag}"',
                f'policy_tag = "{cfg.policy_tag}"',
                f'optimizer = "{cfg.optimizer}"',
                f"num_rounds = {int(num_rounds)}",
                "num_reps = 1",
                "lr = 0.001",
                "sigma = 0.001",
                "population_size = 128",
                "max_tokens = 1024",
                "temperature = 0.0",
                "samples_per_prompt = 1",
                "prompt_batch_size = 2",
                "pass_at_k = false",
                "steps_per_adapter = 4",
                "steps_per_eval = 10",
                "eval_batch_size = 128",
                "use_wandb = false",
                "",
            ]
        )
    )


@cli.command(name="validate", help="Validate an [llm] config without running it.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "-o",
    "--opt",
    "overrides",
    multiple=True,
    help="Override config key: --opt key=value",
)
def validate(config_toml: str, overrides: tuple[str, ...]) -> None:
    cfg = _load_and_parse(config_toml, overrides)
    print(json.dumps(_cfg_summary(cfg), indent=2, sort_keys=True))


@cli.command(name="local", help="Run locally from an [llm] config.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "-o",
    "--opt",
    "overrides",
    multiple=True,
    help="Override config key: --opt key=value",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and print the resolved run, but do not execute.",
)
def local(config_toml: str, overrides: tuple[str, ...], dry_run: bool) -> None:
    cfg = _load_and_parse(config_toml, overrides)
    print(f"EXP_DIR: {cfg.exp_dir}")
    print("LLM:", json.dumps(_cfg_summary(cfg), sort_keys=True))
    if dry_run or cfg.dry_run:
        print("DRY_RUN: true")
        return
    if cfg.optimizer == "eggroll":
        try:
            from llm.eggroll import run_eggroll

            observer = SplitConsoleObserver(log_dir=cfg.exp_dir)
            if sys.stdout.isatty():
                with observer, tee_stdout_to_exp(observer):
                    result = run_eggroll(cfg)
            else:
                result = run_eggroll(cfg)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        print("RESULT:", json.dumps(result, sort_keys=True))
        return
    raise click.ClickException(f"optimizer='{cfg.optimizer}' is parsed but its direct LLM runtime is not wired yet.")


def _load_and_parse(config_toml: str, overrides: tuple[str, ...]) -> LLMConfig:
    try:
        cfg = _load_toml_config(config_toml)
        if overrides:
            cfg = {**cfg, **_parse_overrides(overrides)}
        return _parse_cfg(cfg)
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
