import subprocess
from pathlib import Path

_SETUP_TIMEOUT_SECONDS = 24 * 60 * 60
RUNTIME_HYPERSCALEES = "hyperscalees"
RUNTIME_GEMMA4_MTP = "gemma4-mtp"
DEFAULT_RUNTIME = RUNTIME_HYPERSCALEES
_GEMMA4_MTP_IMAGE = "vllm/vllm-openai:gemma4-0505-cu129"
_VLLM_NIGHTLY_CU129_INDEX = "https://wheels.vllm.ai/nightly/cu129"
_TORCH_CU129_INDEX = "https://download.pytorch.org/whl/cu129"
_GEMMA4_RUNTIME_PACKAGES = (
    "click",
    "datasets==4.4.1",
    "accelerate==1.11.0",
    "peft==0.18.0",
    "safetensors==0.6.2",
    "math-verify[antlr4_9_3]==0.9.0",
    "pylatexenc==2.10",
    "ray==2.51.1",
)
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
)


def _run_hyperscalees_setup():
    subprocess.run(["bash", "admin/setup-hyperscalees.sh", "--skip-verify"], cwd="/root", check=True)


def selected_modal_runtime(argv: list[str] | None = None) -> str:
    return _validate_runtime(_runtime_from_argv(argv or []))


def mk_image(modal, runtime: str = DEFAULT_RUNTIME):
    runtime = _validate_runtime(runtime)
    if runtime == RUNTIME_GEMMA4_MTP:
        return _mk_gemma4_mtp_image(modal)
    return _mk_hyperscalees_image(modal)


def _mk_hyperscalees_image(modal):
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
    return _add_source_mounts(image, project_root)


def _mk_gemma4_mtp_image(modal):
    project_root = Path(__file__).resolve().parents[1]
    image = (
        modal.Image.from_registry(_GEMMA4_MTP_IMAGE, add_python="3.12")
        .entrypoint([])
        .uv_pip_install(
            "vllm",
            pre=True,
            extra_index_url=_VLLM_NIGHTLY_CU129_INDEX,
            extra_options=f"--extra-index-url {_TORCH_CU129_INDEX} --index-strategy unsafe-best-match",
        )
        .pip_install(*_GEMMA4_RUNTIME_PACKAGES)
        .env(
            {
                "PYTHONPATH": "/root",
                "HF_HOME": "/root/.cache/huggingface",
                "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
            }
        )
    )
    return _add_source_mounts(image, project_root)


def _add_source_mounts(image, project_root: Path):
    for d in _SOURCE_DIRS:
        image = image.add_local_dir(str(project_root / d), remote_path=f"/root/{d}")
    return image


def _runtime_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--runtime" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--runtime="):
            return arg.split("=", 1)[1]
    return DEFAULT_RUNTIME


def _validate_runtime(runtime: str) -> str:
    value = str(runtime).strip()
    if value in {RUNTIME_HYPERSCALEES, RUNTIME_GEMMA4_MTP}:
        return value
    raise ValueError(f"Unsupported Modal runtime {runtime!r}. Use {RUNTIME_HYPERSCALEES!r} or {RUNTIME_GEMMA4_MTP!r}.")
