import subprocess
from pathlib import Path

SETUP_TIMEOUT_SECONDS = 24 * 60 * 60
ISAAC_SIM_BASE_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.0-dev2"
PYTHON_VERSION = "3.12"
MAMBA_ROOT_PREFIX = "/opt/conda"


def run_hyperscalees_install() -> None:
    subprocess.run(
        ["bash", "admin/setup-hyperscalees.sh", "--skip-verify"],
        cwd="/root",
        check=True,
    )


def install_micromamba_command() -> str:
    return (
        f"mkdir -p {MAMBA_ROOT_PREFIX} /usr/local/bin && "
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && "
        "micromamba --version"
    )


def mk_hyperscalees_base_image(modal, project_root: Path):
    image = (
        modal.Image.from_registry(ISAAC_SIM_BASE_IMAGE, add_python=PYTHON_VERSION)
        .entrypoint([])
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
        .env(
            {
                "MAMBA_ROOT_PREFIX": MAMBA_ROOT_PREFIX,
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "OMNI_KIT_ACCEPT_EULA": "YES",
            }
        )
        .run_commands(install_micromamba_command())
        .pip_install("modal", "grpclib")  # Required for Modal worker stability
    )
    image = image.add_local_dir(str(project_root / "admin"), remote_path="/root/admin", copy=True)
    return image.run_function(
        run_hyperscalees_install,
        gpu="L4",
        timeout=SETUP_TIMEOUT_SECONDS,
    )
