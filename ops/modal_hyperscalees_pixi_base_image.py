from __future__ import annotations

import shlex
from pathlib import Path

from ops.modal_hyperscalees_base_image import (
    PYTHON_VERSION,
    _bash,
)

PIXI_BASE_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PIXI_HOME = "/opt/pixi"
PIXI_BIN = "/usr/local/bin/pixi"
PIXI_WORKSPACE_DIR = "/opt/yubo-pixi"
PIXI_MANIFEST_PATH = f"{PIXI_WORKSPACE_DIR}/pixi.toml"
PIXI_LOCK_PATH = f"{PIXI_WORKSPACE_DIR}/pixi.lock"
HYPERSCALEES_PIXI_ENV = "hyperscalees"
ISAACLAB_PIXI_ENV = "isaaclab"


def install_pixi_command() -> str:
    return (
        f"mkdir -p {shlex.quote(PIXI_HOME)} /usr/local/bin && "
        "curl -fsSL https://pixi.sh/install.sh "
        f"| PIXI_HOME={shlex.quote(PIXI_HOME)} PIXI_BIN_DIR=/usr/local/bin PIXI_NO_PATH_UPDATE=1 sh && "
        f"{shlex.quote(PIXI_BIN)} --version"
    )


def mk_hyperscalees_pixi_base_image(modal, project_root: Path):
    image = (
        modal.Image.from_registry(PIXI_BASE_IMAGE, add_python=PYTHON_VERSION)
        .entrypoint([])
        .apt_install(
            "bash",
            "build-essential",
            "bzip2",
            "ca-certificates",
            "curl",
            "git",
            "libasound2t64",
            "libdbus-1-3",
            "libegl1",
            "libfontconfig1",
            "libglib2.0-0",
            "libgl1",
            "libgl1-mesa-dev",
            "libglu1-mesa",
            "libice6",
            "libnss3",
            "libsm6",
            "libvulkan1",
            "libx11-6",
            "libx11-dev",
            "libxcursor1",
            "libxcursor-dev",
            "libxext6",
            "libxfixes3",
            "libxi6",
            "libxi-dev",
            "libxinerama1",
            "libxinerama-dev",
            "libxrandr2",
            "libxrandr-dev",
            "libxrender1",
            "libxt6",
            "tar",
            "vulkan-tools",
        )
        .env(
            {
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PIXI_HOME": PIXI_HOME,
            }
        )
        .run_commands(install_pixi_command())
        .pip_install("modal", "grpclib")  # Required for Modal worker stability.
        .run_commands(f"mkdir -p {shlex.quote(PIXI_WORKSPACE_DIR)}")
        .add_local_file(str(project_root / "pixi.toml"), remote_path=PIXI_MANIFEST_PATH, copy=True)
        .add_local_file(str(project_root / "pixi.lock"), remote_path=PIXI_LOCK_PATH, copy=True)
        .run_commands(_pixi_info_command())
        .run_commands(_pixi_install_env_command(ISAACLAB_PIXI_ENV))
        .run_commands(_pixi_task_command(ISAACLAB_PIXI_ENV, "install"))
        .run_commands(_pixi_task_command(ISAACLAB_PIXI_ENV, "check"))
        .run_commands(_pixi_install_env_command(HYPERSCALEES_PIXI_ENV))
        .run_commands(_pixi_task_command(HYPERSCALEES_PIXI_ENV, "setup"))
        .run_commands(_pixi_task_command(HYPERSCALEES_PIXI_ENV, "check"))
    )
    return image


def _pixi_info_command() -> str:
    return _pixi_workspace_command(f"info --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)}")


def _pixi_install_env_command(env_name: str) -> str:
    return _pixi_workspace_command(f"install --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} --locked -e {shlex.quote(env_name)}")


def _pixi_task_command(env_name: str, task_name: str) -> str:
    return _pixi_workspace_command(f"run --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} --locked -e {shlex.quote(env_name)} {shlex.quote(task_name)}")


def _pixi_workspace_command(command: str) -> str:
    return _bash(f"cd {shlex.quote(PIXI_WORKSPACE_DIR)} && {shlex.quote(PIXI_BIN)} {command}")
