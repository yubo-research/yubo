#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

from ops.modal_isaac_render_capture import (
    _official_kit_bare_render_capture_command,
    _official_kit_render_capture_command,
    _official_kit_storm_render_capture_command,
)
from ops.modal_isaac_render_commands import (
    _isaacsim_smoke_command,
    _official_device_caps_command,
    _official_inspect_command,
    _official_isaaclab_rollout_video_command,
    _official_isaacsim_minimal_command,
    _official_isaacsim_smoke_command,
    _official_kit_help_command,
    _official_kit_smoke_command,
    _official_probe_command,
    _probe_command,
)

app = modal.App(name="yubo-isaac-render-probe")

_GPU = os.environ.get("YUBO_RENDER_GPU", "L40S")
_BASE_IMAGE = os.environ.get("YUBO_RENDER_BASE_IMAGE", "debian")
_ISAAC_SIM_IMAGE = os.environ.get("YUBO_RENDER_ISAAC_SIM_IMAGE", "nvcr.io/nvidia/isaac-sim:6.0.0-dev2")
_NVIDIA_DRIVER_VERSION = os.environ.get("YUBO_RENDER_DRIVER_VERSION", "580.95.05")
_TIMEOUT_SECONDS = int(os.environ.get("YUBO_RENDER_TIMEOUT_SECONDS", str(6 * 60 * 60)))

_APT_PACKAGES = (
    "binutils",
    "ca-certificates",
    "curl",
    "gnupg",
    "libegl1",
    "libgl1",
    "libgles2",
    "libglu1-mesa",
    "libvulkan1",
    "libx11-6",
    "libxext6",
    "libxi6",
    "libxrandr2",
    "libxt6",
    "mesa-utils",
    "mesa-utils-extra",
    "vulkan-tools",
)

_NVIDIA_VULKAN_PACKAGES = (
    "libnvidia-cfg1",
    "libnvidia-common",
    "libnvidia-compute",
    "libnvidia-decode",
    "libnvidia-encode",
    "libnvidia-extra",
    "libnvidia-fbc1",
    "libnvidia-gl",
    "libnvidia-gpucomp",
)

_ENV = {
    "ACCEPT_EULA": "Y",
    "NVIDIA_DRIVER_CAPABILITIES": "all",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "PYTHONPATH": "/root",
    "PYTHONUNBUFFERED": "1",
}

_OPS_MOUNT_IGNORE = (
    "__pycache__",
    "**/.DS_Store",
    "**/*.pyc",
    "**/__pycache__",
)


def _base_image() -> modal.Image:
    if _BASE_IMAGE == "debian":
        return modal.Image.debian_slim(python_version="3.12")
    return modal.Image.from_registry(_BASE_IMAGE, add_python="3.12")


def _nvidia_repo_install_command() -> str:
    version = shlex.quote(_NVIDIA_DRIVER_VERSION)
    package_names = " ".join(shlex.quote(name) for name in _NVIDIA_VULKAN_PACKAGES)
    script = (
        "set -euxo pipefail; "
        ". /etc/os-release; "
        'case "${ID}:${VERSION_ID}" in '
        "debian:12*) repo=debian12 ;; "
        "ubuntu:22.04*) repo=ubuntu2204 ;; "
        "ubuntu:24.04*) repo=ubuntu2404 ;; "
        '*) echo "unsupported base OS for NVIDIA CUDA apt repo: ${ID} ${VERSION_ID}" >&2; exit 1 ;; '
        "esac; "
        "rm -f /usr/share/keyrings/nvidia-cuda.gpg /etc/apt/sources.list.d/nvidia-cuda.list; "
        'curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/3bf863cc.pub" '
        "| gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg; "
        'echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] '
        'https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/ /" '
        "> /etc/apt/sources.list.d/nvidia-cuda.list; "
        "apt-get update; "
        f"driver_version={version}; "
        'driver_branch="${driver_version%%.*}"; '
        "install_specs=; "
        f"for base_name in {package_names}; do "
        'package_name="${base_name}-${driver_branch}"; '
        'package_version="$(apt-cache madison "${package_name}" | awk \'{print $3}\' | grep -E "^${driver_version}([-.]|$)" | head -1 || true)"; '
        'if [ -n "${package_version}" ]; then install_specs="${install_specs} ${package_name}=${package_version}"; fi; '
        "done; "
        'nscq_version="$(apt-cache madison libnvidia-nscq | awk \'{print $3}\' | grep -E "^${driver_version}([-.]|$)" | head -1 || true)"; '
        'if [ -n "${nscq_version}" ]; then install_specs="${install_specs} libnvidia-nscq=${nscq_version}"; fi; '
        'echo "NVIDIA_GRAPHICS_INSTALL_SPECS=${install_specs}"; '
        'if [ -n "${install_specs}" ]; then DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${install_specs}; fi; '
        "rm -rf /var/lib/apt/lists/*"
    )
    return "bash -lc " + shlex.quote(script)


def _ops_dir() -> str:
    return str(Path(__file__).resolve().parent)


def _repo_dir() -> str:
    return str(Path(__file__).resolve().parents[1])


def _with_ops_mount(image: modal.Image) -> modal.Image:
    return image.add_local_dir(_ops_dir(), remote_path="/root/ops", ignore=list(_OPS_MOUNT_IGNORE))


def _with_repo_mount(image: modal.Image) -> modal.Image:
    return image.add_local_dir(_repo_dir(), remote_path="/root", ignore=list(_OPS_MOUNT_IGNORE))


image = _base_image().apt_install(*_APT_PACKAGES).pip_install("modal", "grpclib").run_commands(_nvidia_repo_install_command()).env(_ENV)
image = _with_ops_mount(image)

official_isaacsim_image = (
    modal.Image.from_registry(_ISAAC_SIM_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("binutils", "mesa-utils", "vulkan-tools")
    .pip_install("modal", "grpclib", "numpy", "scipy")
    .env(_ENV)
)
official_isaacsim_image = _with_repo_mount(_with_ops_mount(official_isaacsim_image))

official_isaacsim_nvidia_image = (
    modal.Image.from_registry(_ISAAC_SIM_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("binutils", "ca-certificates", "curl", "gnupg", "mesa-utils", "vulkan-tools")
    .run_commands(_nvidia_repo_install_command())
    .pip_install("modal", "grpclib", "numpy", "scipy")
    .env(_ENV)
)
official_isaacsim_nvidia_image = _with_repo_mount(_with_ops_mount(official_isaacsim_nvidia_image))


def _logged_command(cmd: list[str], *, cwd: str = "/root") -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[isaac-render-probe] $ {printable}", flush=True)
    env = os.environ.copy()
    env.update(_ENV)
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
    print(f"[isaac-render-probe] exit={return_code} cmd={printable}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {printable}")
    return return_code


@app.function(image=image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_render_probe(command: str) -> str:
    commands = {
        "probe": _probe_command(),
        "probe-symlink": _probe_command(symlink_nvidia0=True),
        "isaacsim-smoke": _isaacsim_smoke_command(),
    }
    resolved = commands.get(command.strip(), command)
    _logged_command(["bash", "-lc", resolved])
    return "ok"


@app.function(image=official_isaacsim_image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_official_render_probe(command: str) -> str:
    commands = {
        "official-inspect": _official_inspect_command(),
        "official-device-caps": _official_device_caps_command(),
        "official-probe": _official_probe_command(),
        "official-isaacsim-smoke": _official_isaacsim_smoke_command(),
        "official-isaacsim-minimal": _official_isaacsim_minimal_command(),
        "official-isaacsim-minimal-base": _official_isaacsim_minimal_command(experience="/isaac-sim/apps/isaacsim.exp.base.kit"),
        "official-isaacsim-minimal-noview": _official_isaacsim_minimal_command(disable_viewport_updates=True),
        "official-kit-help": _official_kit_help_command(),
        "official-isaaclab-rollout-video": _official_isaaclab_rollout_video_command(),
        "official-kit-smoke": _official_kit_smoke_command(),
        "official-kit-smoke-light": _official_kit_smoke_command(disable_heavy_extensions=True),
        "official-kit-render-capture": _official_kit_render_capture_command(),
        "official-kit-bare-render-capture": _official_kit_bare_render_capture_command(),
        "official-kit-storm-render-capture": _official_kit_storm_render_capture_command(),
    }
    resolved = commands.get(command.strip(), command)
    _logged_command(["bash", "-lc", resolved])
    return "ok"


@app.function(image=official_isaacsim_nvidia_image, gpu=_GPU, timeout=_TIMEOUT_SECONDS)
def run_official_nvidia_render_probe(command: str) -> str:
    commands = {
        "official-nvidia-probe": _official_probe_command(),
        "official-nvidia-device-caps": _official_device_caps_command(),
        "official-nvidia-kit-bare-render-capture": _official_kit_bare_render_capture_command(),
        "official-nvidia-kit-storm-render-capture": _official_kit_storm_render_capture_command(),
        "official-nvidia-isaacsim-minimal": _official_isaacsim_minimal_command(),
    }
    resolved = commands.get(command.strip(), command)
    _logged_command(["bash", "-lc", resolved])
    return "ok"


@app.local_entrypoint()
def main(command: str = "probe") -> None:
    print(f"[isaac-render-probe] gpu={_GPU!r}", flush=True)
    print(f"[isaac-render-probe] base_image={_BASE_IMAGE!r}", flush=True)
    print(f"[isaac-render-probe] isaac_sim_image={_ISAAC_SIM_IMAGE!r}", flush=True)
    print(
        f"[isaac-render-probe] nvidia_driver_version={_NVIDIA_DRIVER_VERSION!r}",
        flush=True,
    )
    print(f"[isaac-render-probe] command={command!r}", flush=True)
    if command.strip().startswith("official-nvidia-"):
        run_official_nvidia_render_probe.remote(command)
    elif command.strip().startswith("official-"):
        run_official_render_probe.remote(command)
    else:
        run_render_probe.remote(command)
