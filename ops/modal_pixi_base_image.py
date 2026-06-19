from __future__ import annotations

import base64
import shlex
from pathlib import Path

# Pixi owns Python + CUDA toolkit (12.8) in the primary Pixi env. Modal hosts the driver.
PYTHON_VERSION = "3.12"
PIXI_HOME = "/opt/pixi"
PIXI_BIN = "/usr/local/bin/pixi"
PIXI_WORKSPACE_DIR = "/opt/yubo-pixi"
PIXI_MANIFEST_PATH = f"{PIXI_WORKSPACE_DIR}/pixi.toml"
PIXI_LOCK_PATH = f"{PIXI_WORKSPACE_DIR}/pixi.lock"
PIXI_ADMIN_DIR = f"{PIXI_WORKSPACE_DIR}/admin"
PIXI_CHECK_PATH = f"{PIXI_ADMIN_DIR}/check_pixi_env.py"
ENN_PATCH_PATH = f"{PIXI_ADMIN_DIR}/patch_enn_failure_tolerance_dim.py"
PRIMARY_PIXI_ENV = "yubo"
ISAACLAB_PIXI_ENV = "isaaclab"
PRIMARY_ENV_PREFIX = f"{PIXI_WORKSPACE_DIR}/.pixi/envs/{PRIMARY_PIXI_ENV}"
ISAACLAB_ENV_PREFIX = f"{PIXI_WORKSPACE_DIR}/.pixi/envs/{ISAACLAB_PIXI_ENV}"
ISAACLAB_SOURCE_DIR = f"{PIXI_WORKSPACE_DIR}/src/IsaacLab"
ISAACLAB_SOURCE_PYTHONPATH = (
    f"{ISAACLAB_SOURCE_DIR}/source/isaaclab",
    f"{ISAACLAB_SOURCE_DIR}/source/isaaclab_assets",
    f"{ISAACLAB_SOURCE_DIR}/source/isaaclab_tasks",
    f"{ISAACLAB_SOURCE_DIR}/source/isaaclab_newton",
    f"{ISAACLAB_SOURCE_DIR}/source/isaaclab_physx",
)
PRIMARY_LD_LIBRARY_PATH = f"{PRIMARY_ENV_PREFIX}/lib:/usr/lib/x86_64-linux-gnu"


def install_pixi_command() -> str:
    return (
        f"mkdir -p {shlex.quote(PIXI_HOME)} /usr/local/bin && "
        "curl -fsSL https://pixi.sh/install.sh "
        f"| PIXI_HOME={shlex.quote(PIXI_HOME)} PIXI_BIN_DIR=/usr/local/bin PIXI_NO_PATH_UPDATE=1 sh && "
        f"{shlex.quote(PIXI_BIN)} --version"
    )


# GUI/X11/Vulkan runtime libs needed by mujoco, dm-control, glfw rendering.
_APT_PACKAGES = (
    "bash",
    "build-essential",
    "ca-certificates",
    "curl",
    "git",
    "git-lfs",
    "patchelf",
    "tar",
    "libopenblas-dev",
    "libopenblas0-pthread",
    "libegl1",
    "libgl1",
    "libgl1-mesa-dev",
    "libglu1-mesa",
    "libvulkan1",
    "vulkan-tools",
    "libglib2.0-0",
    "libsm6",
    "libice6",
    "libxext6",
    "libxrender1",
    "libx11-6",
)


def _isaaclab_install_commands() -> tuple[str, ...]:
    return (
        _pixi_install_env_command(ISAACLAB_PIXI_ENV),
        _pixi_task_command(ISAACLAB_PIXI_ENV, "setup"),
        _pixi_task_command(ISAACLAB_PIXI_ENV, "check"),
        _isaaclab_bootstrap_marker_command(),
    )


def mk_pixi_base_image(modal, project_root: Path):
    """Build the cached Pixi env image (primary Pixi env + IsaacLab).

    Layers (each is a cache point):
      1. base OS + apt + pixi binary       (changes rarely)
      2. pixi.toml + pixi.lock             (changes on dep edits)
      3. primary env install + setup + check
      4. isaaclab install + setup + check + marker

    Runtime ``isaaclab_bootstrap_command()`` is a no-op when layer 4 is present.
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .entrypoint([])
        .apt_install(*_APT_PACKAGES)
        .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "PIXI_HOME": PIXI_HOME})
        .run_commands(install_pixi_command())
        .pip_install("modal", "grpclib")  # Required for Modal worker stability.
        .run_commands(f"mkdir -p {shlex.quote(PIXI_WORKSPACE_DIR)} {shlex.quote(PIXI_ADMIN_DIR)}")
        .add_local_file(str(project_root / "pixi.toml"), remote_path=PIXI_MANIFEST_PATH, copy=True)
        .add_local_file(str(project_root / "pixi.lock"), remote_path=PIXI_LOCK_PATH, copy=True)
        .add_local_file(str(project_root / "admin" / "check_pixi_env.py"), remote_path=PIXI_CHECK_PATH, copy=True)
        .add_local_file(str(project_root / "admin" / "patch_enn_failure_tolerance_dim.py"), remote_path=ENN_PATCH_PATH, copy=True)
        .run_commands(
            _pixi_install_env_command(PRIMARY_PIXI_ENV),
            _pixi_task_command(PRIMARY_PIXI_ENV, "setup"),
            _primary_check_command(),
            *_isaaclab_install_commands(),
        )
    )


_ISAACLAB_BOOTSTRAP_MARKER = f"{PIXI_WORKSPACE_DIR}/.isaaclab_bootstrap_ok"
_ISAACLAB_READY_PY = (
    "import importlib.util; "
    "mods=('isaacsim','isaaclab','isaaclab_tasks'); "
    "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
    "raise SystemExit(0 if not missing else 1)"
)


def _isaaclab_bootstrap_marker_command() -> str:
    return _bash(f"touch {shlex.quote(_ISAACLAB_BOOTSTRAP_MARKER)}")


def isaaclab_bootstrap_command(*, force: bool = False) -> str:
    """Skip IsaacLab install when the image or a warm container is already ready."""
    verify = _pixi_run_command(ISAACLAB_PIXI_ENV, "python -c " + shlex.quote(_ISAACLAB_READY_PY))
    marker = shlex.quote(_ISAACLAB_BOOTSTRAP_MARKER)
    install_chain = " && ".join(_isaaclab_install_commands())
    if force:
        return install_chain
    skip = 'echo "[isaaclab] bootstrap: already installed, skipping" >&2'
    return f"if test -f {marker} && {verify}; then {skip}; else {install_chain}; fi"


def _primary_openblas_env_exports() -> str:
    """Ensure libopenblas is visible for ennbo (patchelf adds DT_NEEDED at build time)."""
    return (
        f"export LD_LIBRARY_PATH={shlex.quote(PRIMARY_LD_LIBRARY_PATH)}:${{LD_LIBRARY_PATH:-}} && "
        'echo "OPENBLAS_PROBE: LD_LIBRARY_PATH=${LD_LIBRARY_PATH} (no LD_PRELOAD)" >&2'
    )


def _primary_check_command() -> str:
    return _bash(f"{_primary_openblas_env_exports()} && {_pixi_task_shell(PRIMARY_PIXI_ENV, 'check')}")


def _pixi_task_shell(env_name: str, task_name: str) -> str:
    return (
        f"cd {shlex.quote(PIXI_WORKSPACE_DIR)} && "
        f"{shlex.quote(PIXI_BIN)} run --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} "
        f"--locked -e {shlex.quote(env_name)} {shlex.quote(task_name)}"
    )


def _pixi_info_command() -> str:
    return _pixi_workspace_command(f"info --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)}")


def _pixi_install_env_command(env_name: str) -> str:
    return _pixi_workspace_command(f"install --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} --locked -e {shlex.quote(env_name)}")


def _pixi_task_command(env_name: str, task_name: str) -> str:
    return _bash(_pixi_task_shell(env_name, task_name))


def _pixi_run_command(env_name: str, command: str) -> str:
    pixi_args = f"run --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} --locked -e {shlex.quote(env_name)} {command}"
    return _pixi_workspace_command(pixi_args)


def _pixi_workspace_command(command: str) -> str:
    return _bash(f"cd {shlex.quote(PIXI_WORKSPACE_DIR)} && {shlex.quote(PIXI_BIN)} {command}")


def _bash(command: str) -> str:
    payload = "set -euxo pipefail\n" + command.strip()
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return f'bash -lc "$(printf %s {shlex.quote(encoded)} | base64 -d)"'
