#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

from ops.modal_hyperscalees_image import mk_image
from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script

app = modal.App(name="yubo-hyperscalees")
image = mk_image(modal)

_TIMEOUT_SECONDS = 24 * 60 * 60
_GPU = os.environ.get("GPU_TYPE", "L40S")
_DEFAULT_VIDEO_GLOB = "runs/**/traces/videos/*.mp4"
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _logged_command(cmd: list[str], *, cwd: str = "/root") -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[modal-hyperscalees] $ {printable}", flush=True)
    env = os.environ.copy()
    env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
    return_code = proc.wait()
    print(f"[modal-hyperscalees] exit={return_code} cmd={printable}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {printable}")
    return return_code


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command(command: str) -> str:
    _logged_command(["bash", "-lc", _runtime_command_script(command)])
    return "ok"


def _artifact_relpath(path: Path, *, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return path.name


def _collect_artifacts(patterns: list[str], *, root: str = "/root") -> list[tuple[str, bytes]]:
    root_path = Path(root)
    seen: set[Path] = set()
    artifacts: list[tuple[str, bytes]] = []
    total_bytes = 0

    for pattern in patterns:
        for path in sorted(root_path.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            data = path.read_bytes()
            total_bytes += len(data)
            if total_bytes > _MAX_ARTIFACT_BYTES:
                raise RuntimeError(f"artifact export exceeds {_MAX_ARTIFACT_BYTES} bytes; use a Modal Volume or narrower --export-glob")
            artifacts.append((_artifact_relpath(path, root=root_path), data))

    print(f"[modal-hyperscalees] collected_artifacts={len(artifacts)}", flush=True)
    for relpath, data in artifacts:
        print(f"[modal-hyperscalees] artifact {relpath} {len(data)} bytes", flush=True)
    return artifacts


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command_export(command: str, artifact_globs: list[str]) -> list[tuple[str, bytes]]:
    _logged_command(["bash", "-lc", _runtime_command_script(command)])
    return _collect_artifacts(artifact_globs)


def _runtime_command_script(command: str) -> str:
    return "\n".join(
        [
            nvidia_vulkan_icd_script(),
            "export LD_LIBRARY_PATH=/opt/conda/envs/yubo-hyperscalees/lib:${LD_LIBRARY_PATH:-}",
            command,
        ]
    )


def _preflight_command() -> str:
    return "set -euxo pipefail; nvidia-smi; micromamba run -n yubo-hyperscalees python -c " + shlex.quote(
        "import sys, torch, vllm; print(sys.version); print('torch', torch.__version__, torch.version.cuda); print('vllm', vllm.__version__)"
    )


def _isaaclab_preflight_command() -> str:
    return "set -euxo pipefail; nvidia-smi || true; micromamba run -n yubo-hyperscalees python -c " + shlex.quote(
        "from problems.isaaclab_env_adapters import isaaclab_default_launcher_kwargs; "
        "import importlib.util; "
        "mods = ['problems.isaaclab_env_adapters', 'isaacsim']; "
        "print({name: importlib.util.find_spec(name) is not None for name in mods}); "
        "print(isaaclab_default_launcher_kwargs())"
    )


def _parse_export_globs(export_glob: str, *, export_videos: bool) -> list[str]:
    patterns = [part.strip() for part in export_glob.split(",") if part.strip()]
    if export_videos and _DEFAULT_VIDEO_GLOB not in patterns:
        patterns.append(_DEFAULT_VIDEO_GLOB)
    return patterns


def _write_artifacts(artifacts: list[tuple[str, bytes]], *, export_dir: str) -> None:
    out_root = Path(export_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for relpath, data in artifacts:
        out_path = out_root / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"[modal-hyperscalees] wrote {out_path} {len(data)} bytes", flush=True)


@app.local_entrypoint()
def main(
    command: str = "preflight",
    export_videos: bool = False,
    export_glob: str = "",
    export_dir: str = "artifacts/modal_hyperscalees",
) -> None:
    if command.strip() == "preflight":
        command = _preflight_command()
    elif command.strip() == "isaaclab-preflight":
        command = _isaaclab_preflight_command()
    print(f"[modal-hyperscalees] gpu={_GPU!r}", flush=True)
    print(f"[modal-hyperscalees] command={command!r}", flush=True)
    artifact_globs = _parse_export_globs(export_glob, export_videos=export_videos)
    if artifact_globs:
        print(f"[modal-hyperscalees] export_globs={artifact_globs!r}", flush=True)
        artifacts = run_hyperscalees_command_export.remote(command, artifact_globs)
        _write_artifacts(artifacts, export_dir=export_dir)
    else:
        run_hyperscalees_command.remote(command)
