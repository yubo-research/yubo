import subprocess
from pathlib import Path

_SETUP_TIMEOUT_SECONDS = 24 * 60 * 60


def _run_hyperscalees_setup():
    subprocess.run(["bash", "admin/setup-hyperscalees.sh", "--skip-verify"], cwd="/root", check=True)


def mk_image(modal):
    image = (
        modal.Image.micromamba(python_version="3.12")
        .apt_install(
            "bash",
            "bzip2",
            "ca-certificates",
            "curl",
            "git",
            "tar",
        )
        .env({"PYTHONPATH": "/root"})
    )
    project_root = Path(__file__).resolve().parents[1]

    image = image.add_local_dir(str(project_root / "admin"), remote_path="/root/admin", copy=True)
    image = image.run_function(
        _run_hyperscalees_setup,
        gpu="L4",
        timeout=_SETUP_TIMEOUT_SECONDS,
    )

    for d in [
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
    ]:
        image = image.add_local_dir(str(project_root / d), remote_path=f"/root/{d}")
    return image
