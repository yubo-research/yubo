#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex

import modal

from ops.modal_command_helpers import collect_artifacts, logged_command, parse_export_globs, write_artifacts
from ops.modal_hyperscalees_image import mk_image

app = modal.App(name="yubo-hyperscalees")
image = mk_image(modal)

_TIMEOUT_SECONDS = 24 * 60 * 60
_GPU = os.environ.get("GPU_TYPE", "L40S")
_LOG_PREFIX = "modal-hyperscalees"


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command(command: str) -> str:
    logged_command(["bash", "-lc", _runtime_command_script(command)], log_prefix=_LOG_PREFIX)
    return "ok"


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command_export(command: str, artifact_globs: list[str]) -> list[tuple[str, bytes]]:
    logged_command(["bash", "-lc", _runtime_command_script(command)], log_prefix=_LOG_PREFIX)
    return collect_artifacts(artifact_globs, log_prefix=_LOG_PREFIX)


def _runtime_command_script(command: str) -> str:
    return "\n".join(
        [
            "export LD_LIBRARY_PATH=/opt/conda/envs/yubo-hyperscalees/lib:${LD_LIBRARY_PATH:-}",
            "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json",
            command,
        ]
    )


def _preflight_command() -> str:
    return "set -euxo pipefail; nvidia-smi; micromamba run -n yubo-hyperscalees python -c " + shlex.quote(
        "import sys, torch, vllm; print(sys.version); print('torch', torch.__version__, torch.version.cuda); print('vllm', vllm.__version__)"
    )


def _vllm_probe_command() -> str:
    return "set -euxo pipefail; nvidia-smi; micromamba run -n yubo-hyperscalees python -c " + shlex.quote(
        "import inspect, vllm; "
        "from llm.vllm_actor_config import sampling_params; "
        "print('vllm', vllm.__version__); "
        "print('SamplingParams', inspect.signature(vllm.SamplingParams)); "
        "print('LLM.generate', inspect.signature(vllm.LLM.generate)); "
        "sp = sampling_params({'temperature': 0.0, 'max_tokens': 1}); "
        "print(type(sp)); "
        "print('has logprobs arg', 'logprobs' in str(inspect.signature(vllm.SamplingParams))); "
        "print('sampling params repr', sp)"
    )


def _isaaclab_preflight_command() -> str:
    return (
        "set -euxo pipefail; nvidia-smi || true; micromamba run -n yubo-isaaclab env LD_LIBRARY_PATH=/opt/conda/envs/yubo-isaaclab/lib python -c "
        + shlex.quote(
            "from problems.isaaclab_env_adapters import isaaclab_default_launcher_kwargs; "
            "import importlib.util; "
            "mods = ['problems.isaaclab_env_adapters', 'isaacsim']; "
            "print({name: importlib.util.find_spec(name) is not None for name in mods}); "
            "print(isaaclab_default_launcher_kwargs())"
        )
    )


@app.local_entrypoint()
def main(
    command: str = "preflight",
    export_videos: bool = False,
    export_glob: str = "",
    export_dir: str = "artifacts/modal_hyperscalees",
) -> None:
    if command.strip() == "preflight":
        command = _preflight_command()
    elif command.strip() == "vllm-probe":
        command = _vllm_probe_command()
    elif command.strip() == "isaaclab-preflight":
        command = _isaaclab_preflight_command()
    print(f"[modal-hyperscalees] gpu={_GPU!r}", flush=True)
    print(f"[modal-hyperscalees] command={command!r}", flush=True)
    artifact_globs = parse_export_globs(export_glob, export_videos=export_videos)
    if artifact_globs:
        print(f"[modal-hyperscalees] export_globs={artifact_globs!r}", flush=True)
        artifacts = run_hyperscalees_command_export.remote(command, artifact_globs)
        write_artifacts(artifacts, export_dir=export_dir, log_prefix=_LOG_PREFIX)
    else:
        run_hyperscalees_command.remote(command)
