import subprocess
from pathlib import Path

_SETUP_TIMEOUT_SECONDS = 24 * 60 * 60

_SOURCE_DIRS = (
    "acq",
    "analysis",
    "common",
    "configs",
    "experiments",
    "llm",
    "model",
    "ops",
    "optimizer",
    "policies",
    "problems",
    "rl",
    "sampling",
    "torch_truncnorm",
    "turbo_m_ref",
    "gym",
    "testing_support",
    "tests",
)
_ROOT_FILES = (
    "sitecustomize.py",
    "pyproject.toml",
    "requirements.txt",
)


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
            "bzip2",
            "ca-certificates",
            "curl",
            "git",
            "tar",
            "build-essential",
        )
        .env({"PYTHONPATH": "/root"})
        .pip_install("modal", "grpclib")  # Required for Modal worker stability
    )

    image = image.add_local_dir(str(project_root / "admin"), remote_path="/root/admin", copy=True)
    image = image.run_function(
        _run_hyperscalees_setup,
        gpu="L4",
        timeout=_SETUP_TIMEOUT_SECONDS,
    )

    return _add_source_mounts(image, project_root)


def _add_source_mounts(image, project_root: Path):
    for d in _SOURCE_DIRS:
        image = image.add_local_dir(str(project_root / d), remote_path=f"/root/{d}")
    for f in _ROOT_FILES:
        image = image.add_local_file(str(project_root / f), remote_path=f"/root/{f}")
    return image
