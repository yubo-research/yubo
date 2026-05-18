import subprocess
from pathlib import Path

_SETUP_TIMEOUT_SECONDS = 24 * 60 * 60

_REPO_MOUNT_IGNORE = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "_tmp",
    "build",
    "dist",
    "node_modules",
    "results",
    "runs",
    "target",
    "venv",
    "**/.DS_Store",
    "**/.ipynb_checkpoints",
    "**/*.egg-info",
    "**/*.pyc",
    "**/__pycache__",
    "**/enn_wasm",
    "**/target",
)
_MODAL_ENV = {
    "NVIDIA_DRIVER_CAPABILITIES": "all",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "PYTHONPATH": "/root",
    "PYTHONUNBUFFERED": "1",
}


def _run_hyperscalees_setup():
    subprocess.run(
        ["bash", "admin/setup-hyperscalees.sh", "--skip-verify"],
        cwd="/root",
        check=True,
    )


def mk_image(modal):
    project_root = Path(__file__).resolve().parents[1]
    image = (
        modal.Image.micromamba(python_version="3.12")
        .apt_install(
            "bash",
            "build-essential",
            "bzip2",
            "ca-certificates",
            "curl",
            "git",
            "libegl1",
            "libgl1",
            "libglu1-mesa",
            "libvulkan1",
            "libxcursor1",
            "libxi6",
            "libxinerama1",
            "libxrandr2",
            "libxt6",
            "tar",
            "vulkan-tools",
        )
        .env(_MODAL_ENV)
        .pip_install("modal", "grpclib")  # Required for Modal worker stability
    )

    image = image.add_local_dir(str(project_root / "admin"), remote_path="/root/admin", copy=True)
    image = image.run_function(
        _run_hyperscalees_setup,
        gpu="L4",
        timeout=_SETUP_TIMEOUT_SECONDS,
    )

    return _add_repo_mount(image, project_root)


def _add_repo_mount(image, project_root: Path):
    return image.add_local_dir(
        str(project_root),
        remote_path="/root",
        ignore=list(_REPO_MOUNT_IGNORE),
    )
