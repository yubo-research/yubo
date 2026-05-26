from pathlib import Path

from ops.modal_hyperscalees_pixi_base_image import mk_hyperscalees_pixi_base_image

_REPO_MOUNT_IGNORE = (
    ".cache",
    ".git",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "_tmp",
    "artifacts",
    "build",
    "dist",
    "node_modules",
    "results",
    "runs",
    "target",
    "venv",
    "videos",
    "wandb",
    "**/.DS_Store",
    "**/.ipynb_checkpoints",
    "**/*.ckpt",
    "**/*.egg-info",
    "**/*.log",
    "**/*.mp4",
    "**/*.pt",
    "**/*.pth",
    "**/*.pyc",
    "**/*.safetensors",
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


def mk_image(modal):
    project_root = Path(__file__).resolve().parents[1]
    image = mk_hyperscalees_pixi_base_image(modal, project_root).env(_MODAL_ENV)
    return _add_repo_mount(image, project_root)


def _add_repo_mount(image, project_root: Path):
    return image.add_local_dir(
        str(project_root),
        remote_path="/root",
        ignore=list(_REPO_MOUNT_IGNORE),
    )
