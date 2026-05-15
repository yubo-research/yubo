#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import tomllib

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LIVE_UPSTREAM_RWKV_CHOICES = {
    "7w0.1B",
    "7w0.4B",
    "7w1.5B",
    "7w3B",
    "7n0.1B",
    "7n0.4B",
    "7n1.5B",
}

_ADAPTER_BLOCKED_ENV_PREFIXES: set[str] = set()

_ADAPTER_BLOCKED_GYMNAX_ENVS: dict[str, str] = {}

_OPTIONAL_RUNTIME_MODULES = {
    "brax:": "brax",
    "craftax:": "craftax",
    "jaxmarl:": "jaxmarl",
    "jumanji:": "jumanji",
    "kinetix:": "kinetix",
    "navix:": "navix",
}


@dataclass
class CoverageValidation:
    path: str
    kind: str
    status: str
    detail: str


@dataclass
class PaperCoverage:
    expected: int
    present: int
    missing: list[str]


@dataclass
class LaunchReadiness:
    path: str
    kind: str
    status: str
    command: str
    detail: str


_EXPECTED_PAPER_CONFIGS = {
    "configs/bo/eggroll/paper/distillation/rwkv_7g7b_int8_gsm8k_distill_eggroll.toml",
    "configs/bo/eggroll/paper/hft/lobs5_360m_goog_2023_pnl_eggroll.toml",
    "configs/bo/eggroll/paper/marl/mpe_simple_reference_v3_eggroll.toml",
    "configs/bo/eggroll/paper/marl/mpe_simple_speaker_listener_v4_eggroll.toml",
    "configs/bo/eggroll/paper/marl/mpe_simple_spread_v3_eggroll.toml",
    "configs/bo/eggroll/paper/qwen/qwen3_1p7b_deepscaler_passk4_eggroll.toml",
    "configs/bo/eggroll/paper/qwen/qwen3_4b_deepscaler_rlvr_eggroll.toml",
    "configs/bo/eggroll/paper/rl/brax_ant_eggroll.toml",
    "configs/bo/eggroll/paper/rl/brax_humanoid_eggroll.toml",
    "configs/bo/eggroll/paper/rl/brax_inverted_double_pendulum_eggroll.toml",
    "configs/bo/eggroll/paper/rl/cartpole_v1_eggroll.toml",
    "configs/bo/eggroll/paper/rl/craftax_classic_symbolic_eggroll.toml",
    "configs/bo/eggroll/paper/rl/craftax_symbolic_eggroll.toml",
    "configs/bo/eggroll/paper/rl/jumanji_game2048_eggroll.toml",
    "configs/bo/eggroll/paper/rl/jumanji_knapsack_eggroll.toml",
    "configs/bo/eggroll/paper/rl/jumanji_snake_eggroll.toml",
    "configs/bo/eggroll/paper/rl/kinetix_hard_pinball_eggroll.toml",
    "configs/bo/eggroll/paper/rl/kinetix_thrust_over_ball_eggroll.toml",
    "configs/bo/eggroll/paper/rl/kinetix_thrustcontrol_left_eggroll.toml",
    "configs/bo/eggroll/paper/rl/navix_doorkey_8x8_eggroll.toml",
    "configs/bo/eggroll/paper/rl/navix_dynamic_obstacles_6x6_random_eggroll.toml",
    "configs/bo/eggroll/paper/rl/navix_fourrooms_eggroll.toml",
    "configs/bo/eggroll/paper/rl/pendulum_v1_eggroll.toml",
    "configs/bo/eggroll/paper/speed/linear_bf16_eggroll.toml",
    "configs/pretrain/hyperscalees/paper/countdownn_7w1p5b_eggroll.toml",
    "configs/pretrain/hyperscalees/paper/countdownn_7w1p5b_grpo.toml",
    "configs/pretrain/hyperscalees/paper/gsm8k_7w3b_eggroll.toml",
    "configs/pretrain/hyperscalees/paper/gsm8k_7w3b_grpo.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop1024_batch16.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop1048576_batch16.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop262144_batch16.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop2_batch16.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop64_batch16.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop65536_batch16.toml",
    "configs/pretrain/nanoegg/paper/minipile_pop8192_batch16.toml",
}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_hyperscalees(path: Path, *, require_live_assets: bool) -> CoverageValidation:
    try:
        from experiments import hyperscalees_llm

        cfg = hyperscalees_llm._load_toml_config(str(path))
        cfg = hyperscalees_llm._finalize_config(cfg)
        cmd = hyperscalees_llm._make_command(cfg["hyperscalees"], cfg["args"])
    except Exception as exc:  # noqa: BLE001 - report validation failures without hiding type.
        return CoverageValidation(_rel(path), "hyperscalees_llm", "invalid", str(exc))

    model_choice = str(cfg["args"].get("model_choice", ""))
    script = str(cfg["hyperscalees"].get("script", ""))
    task = str(cfg["args"].get("task", ""))
    dry_run = bool(cfg["experiment"].get("dry_run", False))

    if model_choice and model_choice not in _LIVE_UPSTREAM_RWKV_CHOICES:
        status = "asset_blocked"
        detail = f"{script} task={task} model_choice={model_choice} dry_run={dry_run}; upstream registry asset not known-live"
        if require_live_assets:
            status = "invalid"
        return CoverageValidation(_rel(path), "hyperscalees_llm", status, detail)

    detail = f"{script} task={task} model_choice={model_choice} dry_run={dry_run}; argv={len(cmd)}"
    return CoverageValidation(_rel(path), "hyperscalees_llm", "ok", detail)


def _validate_nanoegg(path: Path) -> CoverageValidation:
    try:
        from experiments import nanoegg_pretrain

        cfg = nanoegg_pretrain._load_toml_config(str(path))
        cfg = nanoegg_pretrain._finalize_config(cfg)
        cmd = nanoegg_pretrain._make_command(Path(cfg["experiment"]["repo_dir"]), cfg["nanoegg"])
    except Exception as exc:  # noqa: BLE001
        return CoverageValidation(_rel(path), "nanoegg", "invalid", str(exc))

    nanoegg = cfg["nanoegg"]
    detail = (
        f"layers={nanoegg['n_layer']} embd={nanoegg['n_embd']} pop={nanoegg['population_size']} "
        f"batch={nanoegg['batch_size']} epochs={nanoegg['num_epochs']} dry_run={cfg['experiment']['dry_run']}; argv={len(cmd)}"
    )
    return CoverageValidation(_rel(path), "nanoegg", "ok", detail)


def _validate_experiment(path: Path) -> CoverageValidation:
    try:
        from experiments.experiment import load_experiment_config

        cfg = load_experiment_config(config_toml_path=str(path))
    except Exception as exc:  # noqa: BLE001
        return CoverageValidation(_rel(path), "tag_experiment", "invalid", str(exc))

    detail = f"env_tag={cfg.env_tag} policy_tag={cfg.policy_tag} opt_name={cfg.opt_name}"
    adapter_reason = _adapter_block_reason(str(cfg.env_tag))
    if adapter_reason is not None:
        return CoverageValidation(
            _rel(path),
            "tag_experiment",
            "adapter_blocked",
            f"{detail}; {adapter_reason}",
        )
    return CoverageValidation(_rel(path), "tag_experiment", "ok", detail)


def _validate_uhd(path: Path) -> CoverageValidation:
    try:
        from ops.exp_uhd import _load_toml_config, _parse_cfg

        cfg = _parse_cfg(_load_toml_config(str(path)))
    except Exception as exc:  # noqa: BLE001
        return CoverageValidation(_rel(path), "uhd", "invalid", str(exc))

    detail = f"env_tag={cfg.env_tag} policy_tag={cfg.policy_tag} optimizer={cfg.optimizer} num_rounds={cfg.num_rounds} total_timesteps={cfg.total_timesteps}"
    return CoverageValidation(_rel(path), "uhd", "ok", detail)


def _toml_kind(path: Path, default: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except Exception:  # noqa: BLE001 - validation will report the parse error later.
        return default
    if "uhd" in raw:
        return "uhd"
    return default


def _adapter_block_reason(env_tag: str) -> str | None:
    if env_tag in _ADAPTER_BLOCKED_GYMNAX_ENVS:
        return _ADAPTER_BLOCKED_GYMNAX_ENVS[env_tag]
    for prefix in sorted(_ADAPTER_BLOCKED_ENV_PREFIXES):
        if env_tag.startswith(prefix):
            return f"env adapter '{prefix}' is not wired into the Yubo EggRoll runtime yet."
    return None


def _default_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for path in sorted((_PROJECT_ROOT / "configs" / "pretrain" / "hyperscalees").glob("**/*.toml")):
        paths.append(("hyperscalees", path))
    for path in sorted((_PROJECT_ROOT / "configs" / "pretrain" / "nanoegg").glob("**/*.toml")):
        paths.append(("nanoegg", path))
    for path in sorted((_PROJECT_ROOT / "configs" / "bo" / "eggroll").glob("**/*.toml")):
        paths.append((_toml_kind(path, "experiment"), path))
    for path in sorted((_PROJECT_ROOT / "configs" / "bo" / "gymnax").glob("**/*.toml")):
        paths.append((_toml_kind(path, "experiment"), path))
    return paths


def _validate_all(*, require_live_assets: bool) -> list[CoverageValidation]:
    results: list[CoverageValidation] = []
    for kind, path in _default_paths():
        if kind == "hyperscalees":
            results.append(_validate_hyperscalees(path, require_live_assets=require_live_assets))
        elif kind == "nanoegg":
            results.append(_validate_nanoegg(path))
        elif kind == "uhd":
            results.append(_validate_uhd(path))
        elif kind == "experiment":
            results.append(_validate_experiment(path))
        else:
            raise AssertionError(kind)
    return results


def _status_counts(results: list[CoverageValidation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _paper_coverage() -> PaperCoverage:
    present = {path for path in _EXPECTED_PAPER_CONFIGS if (_PROJECT_ROOT / path).is_file()}
    missing = sorted(_EXPECTED_PAPER_CONFIGS - present)
    return PaperCoverage(expected=len(_EXPECTED_PAPER_CONFIGS), present=len(present), missing=missing)


def _launch_command(kind: str, rel_path: str) -> str:
    if kind == "hyperscalees_llm":
        return f"./ops/hyperscalees_llm.py local {rel_path}"
    elif kind == "nanoegg":
        return f"./ops/nanoegg_pretrain.py local {rel_path}"
    elif kind == "uhd":
        return f"./ops/exp_uhd.py local {rel_path}"
    elif kind == "tag_experiment":
        return f"./ops/experiment.py local {rel_path}"
    else:
        raise AssertionError(kind)


def _setup_requirement(kind: str) -> str | None:
    if kind == "hyperscalees_llm" and not (_PROJECT_ROOT / ".external" / "HyperscaleES").is_dir():
        return (
            "HyperscaleES source checkout is required only for upstream "
            "llm_experiments script configs; set [experiment].repo_dir or "
            "provide .external/HyperscaleES."
        )
    if kind == "nanoegg" and not (_PROJECT_ROOT / ".external" / "nano-egg").is_dir():
        return (
            "Legacy nano-egg source-script configs require an explicit "
            ".external/nano-egg checkout. The clean Yubo runtime should port "
            "this behavior instead of importing or shelling through the repo."
        )
    return None


def _dependency_requirement(result: CoverageValidation) -> str | None:
    if result.kind != "tag_experiment" or "env_tag=" not in result.detail:
        return None
    env_tag = result.detail.split("env_tag=", 1)[1].split(" ", 1)[0]
    for prefix, module in sorted(_OPTIONAL_RUNTIME_MODULES.items()):
        if env_tag.startswith(prefix):
            try:
                import importlib.util

                available = importlib.util.find_spec(module) is not None
            except Exception:
                available = False
            if not available:
                return f"Install optional runtime dependency '{module}' in the active Python environment."
    return None


def _readiness_from_validation(result: CoverageValidation) -> LaunchReadiness:
    command = _launch_command(result.kind, result.path)
    if result.status == "invalid":
        return LaunchReadiness(result.path, result.kind, "invalid", command, result.detail)
    if result.status == "asset_blocked":
        return LaunchReadiness(result.path, result.kind, "asset_blocked", command, result.detail)
    if result.status == "adapter_blocked":
        return LaunchReadiness(result.path, result.kind, "adapter_blocked", command, result.detail)

    setup = _setup_requirement(result.kind)
    if setup is not None:
        return LaunchReadiness(
            result.path,
            result.kind,
            "setup_required",
            command,
            f"{result.detail}; {setup}",
        )
    dependency = _dependency_requirement(result)
    if dependency is not None:
        return LaunchReadiness(
            result.path,
            result.kind,
            "dependency_missing",
            command,
            f"{result.detail}; {dependency}",
        )
    return LaunchReadiness(result.path, result.kind, "ready", command, result.detail)


def _readiness_all(*, require_live_assets: bool) -> list[LaunchReadiness]:
    return [_readiness_from_validation(result) for result in _validate_all(require_live_assets=require_live_assets)]


@click.group(help="Inspect and validate EggRoll coverage configs without launching training.")
def cli() -> None:
    pass


@cli.command(help="Validate EggRoll-related TOML configs.")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--require-live-assets",
    is_flag=True,
    help="Treat known stale upstream model choices as invalid instead of asset_blocked.",
)
def validate(json_output: bool, require_live_assets: bool) -> None:
    results = _validate_all(require_live_assets=require_live_assets)
    counts = _status_counts(results)
    paper = _paper_coverage()

    if json_output:
        print(
            json.dumps(
                {
                    "counts": counts,
                    "paper": asdict(paper),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("EggRoll coverage config validation")
        print(
            "counts:",
            ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        )
        print(f"paper coverage: present={paper.present}/{paper.expected}")
        for missing in paper.missing:
            print(f"MISSING       paper_config       {missing}")
        for result in results:
            print(f"{result.status.upper():13} {result.kind:18} {result.path}")
            print(f"  {result.detail}")

    if paper.missing:
        raise click.ClickException("One or more expected EggRoll paper configs are missing.")
    if any(result.status == "invalid" for result in results):
        raise click.ClickException("One or more EggRoll coverage configs are invalid.")


@cli.command(help="Report whether each EggRoll config can actually be launched from this checkout.")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--require-live-assets",
    is_flag=True,
    help="Treat known stale upstream model choices as invalid instead of asset_blocked.",
)
def readiness(json_output: bool, require_live_assets: bool) -> None:
    results = _readiness_all(require_live_assets=require_live_assets)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    if json_output:
        print(
            json.dumps(
                {"counts": counts, "results": [asdict(r) for r in results]},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("EggRoll launch readiness")
        print(
            "counts:",
            ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        )
        for result in results:
            print(f"{result.status.upper():14} {result.kind:18} {result.path}")
            print(f"  command: {result.command}")
            print(f"  detail: {result.detail}")

    if any(result.status == "invalid" for result in results):
        raise click.ClickException("One or more EggRoll configs are invalid.")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
