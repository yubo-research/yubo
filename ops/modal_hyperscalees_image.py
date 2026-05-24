from pathlib import Path

from ops.modal_hyperscalees_base_image import mk_hyperscalees_base_image

_HTTP_RUNTIME_DEPS = (
    "aiohappyeyeballs>=2.5.0",
    "aiosignal>=1.1.2",
    "attrs>=17.3.0",
    "frozenlist>=1.1.1",
    "multidict>=4.5,<7",
    "propcache>=0.2.0",
    "yarl>=1.17,<2",
    "requests>=2.32,<3",
    "idna>=2.5,<4",
    "urllib3>=1.21.1,<3",
    "charset-normalizer>=2,<4",
    "chardet>=3,<6",
    "certifi",
    "grpclib>=0.4.7,<0.5",
)

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
    image = mk_hyperscalees_base_image(modal, project_root).run_commands(_http_runtime_repair_command()).env(_MODAL_ENV)
    return _add_repo_mount(image, project_root)


def _http_runtime_repair_command() -> str:
    deps = " ".join(f"'{dep}'" for dep in _HTTP_RUNTIME_DEPS)
    return f"micromamba run -n yubo-hyperscalees python -m pip install --disable-pip-version-check {deps}"


def _add_repo_mount(image, project_root: Path):
    return image.add_local_dir(
        str(project_root),
        remote_path="/root",
        ignore=list(_REPO_MOUNT_IGNORE),
    )
