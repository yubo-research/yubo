#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

if sys.version_info < (3, 11):
    sys.stderr.write("ops/llm_architecture.py requires Python >= 3.11. Run it from the selected Pixi env.\n")
    raise SystemExit(1)


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_path()

from llm.architecture import discover_architecture_profile, make_update_program  # noqa: E402
from llm.lora import select_lora_target_modules, unsupported_vllm_dense_update_modules  # noqa: E402
from llm.registry import resolve_llm_policy  # noqa: E402


def _load_empty_causal_lm(model_name: str, *, local_files_only: bool) -> Any:
    missing = [module for module in ("accelerate", "torch", "transformers") if _missing_module(module)]
    if missing:
        raise click.ClickException(f"Architecture inspection requires {missing}. Run from the yubo Pixi env.")
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise click.ClickException("Architecture inspection requires accelerate, torch, and transformers.") from exc

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=bool(local_files_only))
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)


def _missing_module(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is None


def _resolve_policy_or_model(value: str, rank: int | None) -> tuple[str, str, int]:
    try:
        policy = resolve_llm_policy(value)
    except KeyError:
        return value, value, 1 if rank is None else int(rank)
    return policy.policy_tag, policy.model_name, int(policy.lora_rank if rank is None else rank)


def inspect_loaded_model(
    *,
    policy_label: str,
    model_name: str,
    model: Any,
    rank: int,
    roles: Any | None,
    layer_band: str,
    expert_policy: str,
    max_targets: int | None,
    direct_vllm_dense_update: bool,
) -> dict[str, Any]:
    program = make_update_program(
        roles=roles,
        layer_band=layer_band,
        expert_policy=expert_policy,
        rank=int(rank),
        max_targets=max_targets,
    )
    profile = discover_architecture_profile(model)
    role_counts = {role: count for role, count in profile.role_counts().items() if count > 0}
    selected_modules = select_lora_target_modules(model_name=model_name, base_model=model, update_program=program)
    unsupported = unsupported_vllm_dense_update_modules(selected_modules) if direct_vllm_dense_update else ()
    return {
        "policy": policy_label,
        "model_name": model_name,
        "model_class": profile.model_class,
        "program": {
            "roles": list(program.roles),
            "layer_band": program.layer_band,
            "expert_policy": program.expert_policy,
            "rank": int(program.rank),
            "scale": float(program.scale),
            "seed": int(program.seed),
            "max_targets": program.max_targets,
        },
        "role_counts": role_counts,
        "num_selected_modules": len(selected_modules),
        "selected_modules": selected_modules,
        "direct_vllm_dense_update": {
            "checked": bool(direct_vllm_dense_update),
            "supported": bool(direct_vllm_dense_update and not unsupported),
            "unsupported_modules": list(unsupported),
        },
    }


def _render_text(report: dict[str, Any], *, limit: int) -> str:
    lines = [
        f"policy: {report['policy']}",
        f"model: {report['model_name']}",
        f"model_class: {report['model_class']}",
        "program:",
        f"  roles: {', '.join(report['program']['roles'])}",
        f"  layer_band: {report['program']['layer_band']}",
        f"  expert_policy: {report['program']['expert_policy']}",
        f"  rank: {report['program']['rank']}",
        f"  max_targets: {report['program']['max_targets']}",
        "roles:",
    ]
    for role, count in sorted(report["role_counts"].items()):
        lines.append(f"  {role}: {count}")
    selected = report["selected_modules"]
    lines.append(f"selected_modules: {len(selected)}")
    for name in selected[: int(limit)]:
        lines.append(f"  {name}")
    if len(selected) > int(limit):
        lines.append(f"  ... {len(selected) - int(limit)} more")
    direct = report["direct_vllm_dense_update"]
    if not direct["checked"]:
        lines.append("direct_vllm_dense_update: not_checked")
    else:
        lines.append(f"direct_vllm_dense_update: {'supported' if direct['supported'] else 'unsupported'}")
        for name in direct["unsupported_modules"][: int(limit)]:
            lines.append(f"  unsupported: {name}")
    return "\n".join(lines)


def _load_report_from_config(path: str, *, local_files_only: bool, direct_vllm_dense_update: bool) -> dict[str, Any]:
    import tomllib

    with open(path, "rb") as f:
        raw = tomllib.load(f)
    if "llm" in raw:
        from experiments.llm import _load_toml_config, _parse_cfg

        cfg = _parse_cfg(_load_toml_config(path))
        rank = cfg.policy.lora_rank
        model = _load_empty_causal_lm(cfg.policy.model_name, local_files_only=local_files_only)
        return inspect_loaded_model(
            policy_label=cfg.policy_tag,
            model_name=cfg.policy.model_name,
            model=model,
            rank=rank,
            roles=cfg.llm_update_roles,
            layer_band=cfg.llm_update_layer_band,
            expert_policy=cfg.llm_update_expert_policy,
            max_targets=cfg.llm_update_max_targets,
            direct_vllm_dense_update=direct_vllm_dense_update,
        )
    from ops.exp_uhd_parse import _load_toml_config, _parse_cfg

    cfg = _parse_cfg(_load_toml_config(path))
    if cfg.policy_tag is None or not str(cfg.env_tag).startswith("llm:"):
        raise click.ClickException("inspect-config currently requires an LLM [uhd] config with policy_tag.")
    _label, model_name, rank = _resolve_policy_or_model(cfg.policy_tag, None)
    model = _load_empty_causal_lm(model_name, local_files_only=local_files_only)
    return inspect_loaded_model(
        policy_label=str(cfg.policy_tag),
        model_name=model_name,
        model=model,
        rank=rank,
        roles=cfg.llm_update_roles,
        layer_band=cfg.llm_update_layer_band,
        expert_policy=cfg.llm_update_expert_policy,
        max_targets=cfg.llm_update_max_targets,
        direct_vllm_dense_update=direct_vllm_dense_update,
    )


def _emit_report(report: dict[str, Any], *, output_format: str, limit: int) -> None:
    if output_format == "json":
        click.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    click.echo(_render_text(report, limit=int(limit)))


@click.group(help="Inspect semantic LLM update targets without launching vLLM engines.")
def cli() -> None:
    pass


def inspect(
    policy_or_model: str,
    roles: str | None,
    layer_band: str,
    expert_policy: str,
    max_targets: int | None,
    rank: int | None,
    local_files_only: bool,
    no_direct_vllm_dense_update: bool,
    output_format: str,
    limit: int,
) -> None:
    policy_label, model_name, resolved_rank = _resolve_policy_or_model(policy_or_model, rank)
    model = _load_empty_causal_lm(model_name, local_files_only=local_files_only)
    report = inspect_loaded_model(
        policy_label=policy_label,
        model_name=model_name,
        model=model,
        rank=resolved_rank,
        roles=roles,
        layer_band=layer_band,
        expert_policy=expert_policy,
        max_targets=max_targets,
        direct_vllm_dense_update=not bool(no_direct_vllm_dense_update),
    )
    _emit_report(report, output_format=output_format, limit=int(limit))


def inspect_config(config_path: str, local_files_only: bool, no_direct_vllm_dense_update: bool, output_format: str, limit: int) -> None:
    report = _load_report_from_config(
        config_path,
        local_files_only=local_files_only,
        direct_vllm_dense_update=not bool(no_direct_vllm_dense_update),
    )
    _emit_report(report, output_format=output_format, limit=int(limit))


inspect = click.option("--limit", default=40, show_default=True, type=int)(inspect)
inspect = click.option("--format", "output_format", default="text", show_default=True, type=click.Choice(["text", "json"]))(inspect)
inspect = click.option("--no-direct-vllm-dense-update", is_flag=True, help="Skip direct vLLM dense-update support reporting.")(inspect)
inspect = click.option("--local-files-only", is_flag=True, help="Require Hugging Face files to exist in the local cache.")(inspect)
inspect = click.option("--rank", default=None, type=int, help="LoRA rank for raw model names; policy tags use their registered rank by default.")(inspect)
inspect = click.option("--max-targets", default=None, type=int)(inspect)
inspect = click.option("--expert-policy", default="all", show_default=True, type=click.Choice(["all", "dense", "routed", "shared", "router"]))(inspect)
inspect = click.option("--layer-band", default="all", show_default=True, type=click.Choice(["all", "early", "middle", "late"]))(inspect)
inspect = click.option("--roles", default=None, help="Comma-separated semantic roles. Defaults to LoRA update roles.")(inspect)
inspect = click.argument("policy_or_model")(inspect)
inspect = cli.command(help="Inspect a policy tag or raw Hugging Face model name.")(inspect)

inspect_config = click.option("--limit", default=40, show_default=True, type=int)(inspect_config)
inspect_config = click.option("--format", "output_format", default="text", show_default=True, type=click.Choice(["text", "json"]))(inspect_config)
inspect_config = click.option("--no-direct-vllm-dense-update", is_flag=True, help="Skip direct vLLM dense-update support reporting.")(inspect_config)
inspect_config = click.option("--local-files-only", is_flag=True, help="Require Hugging Face files to exist in the local cache.")(inspect_config)
inspect_config = click.argument("config_path")(inspect_config)
inspect_config = cli.command(name="inspect-config", help="Inspect the semantic update targets selected by a [llm] or LLM [uhd] TOML config.")(inspect_config)


def main() -> None:
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
