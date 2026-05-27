#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
from pathlib import Path

import modal
import tomllib

from ops.modal_command_helpers import (
    collect_artifacts,
    logged_command,
    parse_export_globs,
    write_artifacts,
)
from ops.modal_hyperscalees_pixi_base_image import (
    HYPERSCALEES_PIXI_ENV,
    ISAACLAB_PIXI_ENV,
    PIXI_BIN,
    PIXI_HOME,
    PIXI_MANIFEST_PATH,
)
from ops.modal_hyperscalees_pixi_image import mk_image
from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script


def _project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1],
        Path("/root"),
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "pixi.toml").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


app = modal.App(name="yubo-hyperscalees-pixi")
image = mk_image(modal)

_TIMEOUT_SECONDS = 24 * 60 * 60
_GPU = os.environ.get("GPU_TYPE", "L40S")
_LOG_PREFIX = "modal-hyperscalees-pixi"
_VALID_PIXI_ENVS = {HYPERSCALEES_PIXI_ENV, ISAACLAB_PIXI_ENV}
_AUTO_PIXI_ENV = "auto"
_REPO_ROOT = _project_root()
_ROOT_EXPERIMENT_KEYS = {"opt_name"}
_ROOT_UHD_KEYS = {"env_tag", "num_rounds", "optimizer", "perturb"}


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command(command: str, env_name: str = HYPERSCALEES_PIXI_ENV) -> str:
    _run_command(command, env_name)
    return "ok"


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command_export(
    command: str,
    artifact_globs: list[str],
    env_name: str = HYPERSCALEES_PIXI_ENV,
) -> list[tuple[str, bytes]]:
    _run_command(command, env_name)
    return collect_artifacts(artifact_globs, log_prefix=_LOG_PREFIX)


def _run_command(command: str, env_name: str) -> None:
    logged_command(
        ["bash", "-lc", _runtime_command_script(command, env_name)],
        log_prefix=_LOG_PREFIX,
        extra_env={"PIXI_HOME": PIXI_HOME},
    )


def _runtime_command_script(command: str, env_name: str) -> str:
    return "\n".join(
        [
            f"export PIXI_HOME={shlex.quote(PIXI_HOME)}",
            _env_prefix_export(env_name, "YUBO_PIXI_PREFIX"),
            "export LD_LIBRARY_PATH=${YUBO_PIXI_PREFIX}/lib:${LD_LIBRARY_PATH:-}",
            nvidia_vulkan_icd_script(),
            command,
        ]
    )


def _pixi_run(env_name: str, command: str) -> str:
    return f"{shlex.quote(PIXI_BIN)} run --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} --locked -e {shlex.quote(env_name)} {command}"


def _env_prefix_export(env_name: str, var_name: str) -> str:
    script = "import sys; print(sys.prefix)"
    return f'export {var_name}="$({_pixi_run(env_name, "python -c " + shlex.quote(script))})"'


def _preflight_command() -> str:
    return "set -euxo pipefail; nvidia-smi; " + _pixi_run(
        HYPERSCALEES_PIXI_ENV,
        "python -c "
        + shlex.quote("import sys, torch, vllm; print(sys.version); print('torch', torch.__version__, torch.version.cuda); print('vllm', vllm.__version__)"),
    )


def _vllm_probe_command() -> str:
    return "set -euxo pipefail; nvidia-smi; " + _pixi_run(
        HYPERSCALEES_PIXI_ENV,
        "python -c "
        + shlex.quote(
            "import inspect, vllm; "
            "from llm.vllm_actor_config import sampling_params; "
            "print('vllm', vllm.__version__); "
            "print('SamplingParams', inspect.signature(vllm.SamplingParams)); "
            "print('LLM.generate', inspect.signature(vllm.LLM.generate)); "
            "sp = sampling_params({'temperature': 0.0, 'max_tokens': 1}); "
            "print(type(sp)); "
            "print('has logprobs arg', 'logprobs' in str(inspect.signature(vllm.SamplingParams))); "
            "print('sampling params repr', sp)"
        ),
    )


def _isaaclab_preflight_command() -> str:
    return "set -euxo pipefail; nvidia-smi || true; " + _pixi_run(
        ISAACLAB_PIXI_ENV,
        "python -c "
        + shlex.quote(
            "from problems.isaaclab_env_adapters import isaaclab_default_launcher_kwargs; "
            "import importlib.util; "
            "mods = ['problems.isaaclab_env_adapters', 'isaacsim']; "
            "print({name: importlib.util.find_spec(name) is not None for name in mods}); "
            "print(isaaclab_default_launcher_kwargs())"
        ),
    )


def _experiment_command(env_name: str, mode: str, config_path: str) -> str:
    runner = _config_runner(config_path, mode)
    return "set -euxo pipefail; cd /root; " + _pixi_run(
        env_name,
        runner,
    )


def _pytest_command(env_name: str, args: str) -> str:
    return "set -euxo pipefail; cd /root; " + _pixi_run(
        env_name,
        f"python -m pytest {args}",
    )


@app.local_entrypoint()
def main(
    command: str = "preflight",
    config: str = "",
    pytest: bool = False,
    pytest_args: str = "-sv tests -rs",
    pixi_env: str = _AUTO_PIXI_ENV,
    experiment_mode: str = "local",
    export_videos: bool = False,
    export_glob: str = "",
    export_dir: str = "artifacts/modal_hyperscalees_pixi",
) -> None:
    config = config.strip()
    if config:
        pixi_env = _resolve_pixi_env(pixi_env, config)
        command = _experiment_command(pixi_env, experiment_mode, config)
    elif pytest:
        pixi_env = _resolve_pixi_env(pixi_env, "")
        command = _pytest_command(pixi_env, pytest_args)
    elif command.strip() == "preflight":
        command = _preflight_command()
        pixi_env = HYPERSCALEES_PIXI_ENV
    elif command.strip() == "vllm-probe":
        command = _vllm_probe_command()
        pixi_env = HYPERSCALEES_PIXI_ENV
    elif command.strip() == "isaaclab-preflight":
        command = _isaaclab_preflight_command()
        pixi_env = ISAACLAB_PIXI_ENV
    else:
        pixi_env = _resolve_pixi_env(pixi_env, "")
    _check_pixi_env(pixi_env)
    print(f"[modal-hyperscalees-pixi] gpu={_GPU!r}", flush=True)
    print(f"[modal-hyperscalees-pixi] pixi_env={pixi_env!r}", flush=True)
    if config:
        print(f"[modal-hyperscalees-pixi] config={config!r}", flush=True)
    if pytest:
        print(f"[modal-hyperscalees-pixi] pytest_args={pytest_args!r}", flush=True)
    print(f"[modal-hyperscalees-pixi] command={command!r}", flush=True)
    artifact_globs = parse_export_globs(export_glob, export_videos=export_videos)
    if artifact_globs:
        print(f"[modal-hyperscalees-pixi] export_globs={artifact_globs!r}", flush=True)
        artifacts = run_hyperscalees_command_export.remote(command, artifact_globs, pixi_env)
        write_artifacts(artifacts, export_dir=export_dir, log_prefix=_LOG_PREFIX)
    else:
        run_hyperscalees_command.remote(command, pixi_env)


def _check_pixi_env(env_name: str) -> None:
    if env_name not in _VALID_PIXI_ENVS:
        valid = ", ".join(sorted(_VALID_PIXI_ENVS))
        raise ValueError(f"Unknown Pixi env {env_name!r}. Valid envs: {valid}")


def _resolve_pixi_env(env_name: str, config_path: str) -> str:
    if env_name != _AUTO_PIXI_ENV:
        return env_name
    if _is_isaaclab_config(config_path):
        return ISAACLAB_PIXI_ENV
    return HYPERSCALEES_PIXI_ENV


def _is_isaaclab_config(config_path: str) -> bool:
    parts = config_path.replace("\\", "/").split("/")
    return "isaaclab" in parts


def _config_runner(config_path: str, mode: str) -> str:
    schema = _config_schema(config_path)
    if schema == "uhd":
        return f"./ops/exp_uhd.py {shlex.quote(mode)} {shlex.quote(config_path)}"
    if schema == "experiment":
        return f"./ops/experiment.py {shlex.quote(mode)} {shlex.quote(config_path)}"
    if schema == "llm":
        return f"./ops/llm.py {shlex.quote(mode)} {shlex.quote(config_path)}"
    if schema == "rl":
        return f"./ops/rl.py {shlex.quote(mode)} {shlex.quote(config_path)}"
    raise ValueError(f"Unsupported config schema {schema!r} in {config_path!r}.")


def _config_schema(config_path: str) -> str:
    with (_REPO_ROOT / config_path).open("rb") as f:
        data = tomllib.load(f)
    if "uhd" in data:
        return "uhd"
    if "experiment" in data:
        return "experiment"
    if "llm" in data:
        return "llm"
    if "rl" in data:
        return "rl"
    if _ROOT_EXPERIMENT_KEYS & set(data):
        return "experiment"
    if _ROOT_UHD_KEYS <= set(data):
        return "uhd"
    return "unknown"
