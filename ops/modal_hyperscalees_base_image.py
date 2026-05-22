from __future__ import annotations

import base64
import shlex
from pathlib import Path

ISAAC_SIM_BASE_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.0-dev2"
PYTHON_VERSION = "3.12"
MAMBA_ROOT_PREFIX = "/opt/conda"
HYPERSCALEES_ENV_NAME = "yubo-hyperscalees"
ISAACLAB_ENV_NAME = "yubo-isaaclab"
ENV_NAME = HYPERSCALEES_ENV_NAME
HF_HOME_DIR = "/root/.cache/yubo/hyperscalees"

HYPERSCALEES_REPO_URL = "https://github.com/ESHyperscale/HyperscaleES.git"
HYPERSCALEES_REPO_COMMIT = "b77f7d6f91238fd575313e946b9cad21e0a74b32"
VECCHIABO_SPEC = "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"
LASSOBENCH_SPEC = "LassoBench @ git+https://github.com/ksehic/LassoBench.git"
PUFFERLIB_SPEC = "pufferlib==3.0.0"

PYTORCH_CU128_INDEX_URL = "https://download.pytorch.org/whl/cu128"
ISAACLAB_SOURCE_URL = "https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_SOURCE_REF = "v3.0.0-beta"
ISAACLAB_SOURCE_DIR = "/root/.cache/yubo/isaaclab/IsaacLab"
ISAACLAB_SOURCE_INSTALL_TARGET = "assets,tasks,newton"

HYPERSCALEES_FINAL_REQUIREMENT_NAMES = ("setuptools", "numpy", "numba", "llvmlite", "protobuf", "dm-control", "mujoco")
HYPERSCALEES_FINAL_NODEPS_REQUIREMENT_NAMES = ("brax", "mujoco-mjx", "mujoco-warp", "warp-lang")
FAISS_OPENBLAS_CONDA_SPECS = (
    "nomkl",
    "libblas=*=*openblas",
    "openblas",
    "libfaiss=1.10.0=cpu_openblas*",
    "faiss=1.10.0=cpu_openblas_py312*",
    "faiss-cpu=1.10.0",
)


def install_micromamba_command() -> str:
    return (
        f"mkdir -p {MAMBA_ROOT_PREFIX} /usr/local/bin && "
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && "
        "micromamba --version"
    )


def mk_hyperscalees_base_image(modal, project_root: Path):
    conda_isaaclab_yml = _read_project_file(project_root, "admin/conda-isaaclab.yml")
    conda_hyperscalees_yml = _read_project_file(project_root, "admin/conda-hyperscalees.yml")
    isaaclab_requirements = _read_project_file(project_root, "admin/requirements-isaaclab.txt")
    hyperscalees_requirements = _read_project_file(project_root, "admin/requirements-hyperscalees.txt")
    hyperscalees_no_deps_requirements = _read_project_file(project_root, "admin/requirements-hyperscalees-nodeps.txt")
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
        .pip_install("modal", "grpclib")  # Required for Modal worker stability.
        .run_commands(_create_conda_env_command(conda_isaaclab_yml, env_name=ISAACLAB_ENV_NAME))
        .run_commands(_install_requirements_command(ISAACLAB_ENV_NAME, isaaclab_requirements, "yubo-requirements-isaaclab.txt"))
        .run_commands(_install_isaaclab_source_command())
        .run_commands(
            _install_requirements_command(
                ISAACLAB_ENV_NAME,
                isaaclab_requirements,
                "yubo-requirements-isaaclab.txt",
                pip_args="--no-deps",
            )
        )
        .run_commands(_validate_isaaclab_runtime_command())
        .run_commands(_create_conda_env_command(conda_hyperscalees_yml))
        .run_commands(_install_python_requirements_command(hyperscalees_requirements, hyperscalees_no_deps_requirements))
        .run_commands(_install_source_extras_command())
        .run_commands(_finalize_runtime_compat_command(hyperscalees_requirements, hyperscalees_no_deps_requirements))
        .run_commands("micromamba run -n yubo-hyperscalees pip install tiktoken rustbpe kernels 'huggingface-hub<1.0'")
    )
    return image


def _read_project_file(project_root: Path, relpath: str) -> str:
    return (project_root / relpath).read_text(encoding="utf-8")


def _create_conda_env_command(conda_yml: str, *, env_name: str = ENV_NAME) -> str:
    tmp_name = f"/tmp/yubo-conda-{env_name.removeprefix('yubo-')}.yml"
    return _bash(
        "\n".join(
            [
                _write_heredoc(tmp_name, conda_yml),
                f"micromamba env create -y -n {shlex.quote(env_name)} -f {shlex.quote(tmp_name)}",
                _env_helpers(env_name),
                "run_in_env python -c 'import sys; print(sys.prefix)'",
            ]
        )
    )


def _install_python_requirements_command(requirements: str, no_deps_requirements: str) -> str:
    return _bash(
        "\n".join(
            [
                _write_heredoc("/tmp/yubo-requirements-hyperscalees.txt", requirements),
                _write_heredoc("/tmp/yubo-requirements-hyperscalees-nodeps.txt", no_deps_requirements),
                _env_helpers(ENV_NAME),
                "pip_install -r /tmp/yubo-requirements-hyperscalees.txt",
                "pip_install --no-deps -r /tmp/yubo-requirements-hyperscalees-nodeps.txt",
            ]
        )
    )


def _install_requirements_command(env_name: str, requirements: str, tmp_name: str, *, pip_args: str = "") -> str:
    pip_prefix = f"{pip_args.strip()} " if pip_args.strip() else ""
    return _bash(
        "\n".join(
            [
                _write_heredoc(f"/tmp/{tmp_name}", requirements),
                _env_helpers(env_name),
                f"pip_install {pip_prefix}-r /tmp/{shlex.quote(tmp_name)}",
            ]
        )
    )


def _install_isaaclab_source_command() -> str:
    return _bash(
        f"""
{_env_helpers(ISAACLAB_ENV_NAME)}
mkdir -p {shlex.quote(str(Path(ISAACLAB_SOURCE_DIR).parent))}
if [ -d {shlex.quote(ISAACLAB_SOURCE_DIR)}/.git ]; then
  git -C {shlex.quote(ISAACLAB_SOURCE_DIR)} fetch --tags origin
else
  git clone {shlex.quote(ISAACLAB_SOURCE_URL)} {shlex.quote(ISAACLAB_SOURCE_DIR)}
fi
git -C {shlex.quote(ISAACLAB_SOURCE_DIR)} checkout -q {shlex.quote(ISAACLAB_SOURCE_REF)}
run_in_env bash -c 'cd {shlex.quote(ISAACLAB_SOURCE_DIR)} && ./isaaclab.sh --install {shlex.quote(ISAACLAB_SOURCE_INSTALL_TARGET)}'
"""
    )


def _install_source_extras_command() -> str:
    return _bash(
        f"""
{_env_helpers(ENV_NAME)}

build_root="$(mktemp -d)"
source_dir="${{build_root}}/HyperscaleES"
wheel_dir="${{build_root}}/wheelhouse"
cleanup() {{
  rm -rf "${{build_root}}"
}}
trap cleanup EXIT
mkdir -p "${{wheel_dir}}"
git init -q "${{source_dir}}"
git -C "${{source_dir}}" remote add origin {shlex.quote(HYPERSCALEES_REPO_URL)}
git -C "${{source_dir}}" fetch -q --depth 1 origin {shlex.quote(HYPERSCALEES_REPO_COMMIT)}
git -C "${{source_dir}}" checkout -q FETCH_HEAD
run_in_env python - "${{source_dir}}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pkg = root / "src" / "hyperscalees"
init_file = pkg / "__init__.py"
models_init = pkg / "models" / "__init__.py"
pyproject = root / "pyproject.toml"
vocab = pkg / "tok_files" / "rwkv_vocab_v20230424.txt"
if not vocab.exists():
    raise SystemExit(f"missing {{vocab}}")
init_text = init_file.read_text(encoding="utf-8")
version_match = re.search(r'__version__\\s*=\\s*["\\']([^"\\']+)["\\']', init_text)
init_file.write_text(f'__version__ = "{{version_match.group(1)}}"\\n', encoding="utf-8")
models_init.write_text("", encoding="utf-8")
pyproject_text = pyproject.read_text(encoding="utf-8")
pyproject.write_text(pyproject_text.replace('jaxrwkv = ["tok_files/*"]', 'hyperscalees = ["tok_files/*"]'), encoding="utf-8")
PY
run_in_env python -m pip wheel --disable-pip-version-check --no-deps --wheel-dir "${{wheel_dir}}" "${{source_dir}}"
wheel="$(find "${{wheel_dir}}" -maxdepth 1 -name 'hyperscalees-*.whl' -print -quit)"
pip_install --no-deps "${{wheel}}"

{_faiss_openblas_repair_command(ENV_NAME)}
run_in_env bash -c '
  LDFLAGS="-L${{CONDA_PREFIX}}/lib" \\
  LIBRARY_PATH="${{CONDA_PREFIX}}/lib" \\
  CPATH="$(python -c "import pybind11; print(pybind11.get_include())")" \\
  python -m pip install --disable-pip-version-check --no-build-isolation --no-deps "{VECCHIABO_SPEC}"
'
pip_install --no-deps {shlex.quote(LASSOBENCH_SPEC)}

cuda_version="$(run_in_env python - <<'PY'
import torch

cuda_version = torch.version.cuda
if not cuda_version:
    raise SystemExit("torch.version.cuda is empty")
print(".".join(cuda_version.split(".")[:2]))
PY
)"
micromamba install -y -n {shlex.quote(ENV_NAME)} -c nvidia -c conda-forge \\
  "cuda-toolkit=${{cuda_version}}" "cuda-nvcc=${{cuda_version}}" "cuda-cudart-dev=${{cuda_version}}" \\
  ninja cmake gxx_linux-64
run_in_env bash -c '
  export CUDA_HOME="${{CONDA_PREFIX}}"
  export CUDACXX="${{CONDA_PREFIX}}/bin/nvcc"
  export TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST:-8.9}}"
  python -m pip install --disable-pip-version-check --no-build-isolation --no-deps "{PUFFERLIB_SPEC}"
'
"""
    )


def _finalize_runtime_compat_command(requirements: str | None = None, no_deps_requirements: str | None = None) -> str:
    requirements = requirements if requirements is not None else _read_repo_file("admin/requirements-hyperscalees.txt")
    no_deps_requirements = no_deps_requirements if no_deps_requirements is not None else _read_repo_file("admin/requirements-hyperscalees-nodeps.txt")
    return _bash(
        "\n".join(
            [
                _write_heredoc("/tmp/yubo-requirements-hyperscalees.txt", requirements),
                _write_heredoc("/tmp/yubo-requirements-hyperscalees-nodeps.txt", no_deps_requirements),
                _env_helpers(ENV_NAME),
                _select_requirements_command(
                    "/tmp/yubo-requirements-hyperscalees.txt",
                    "/tmp/yubo-requirements-hyperscalees-final.txt",
                    HYPERSCALEES_FINAL_REQUIREMENT_NAMES,
                ),
                _select_requirements_command(
                    "/tmp/yubo-requirements-hyperscalees-nodeps.txt",
                    "/tmp/yubo-requirements-hyperscalees-final-nodeps.txt",
                    HYPERSCALEES_FINAL_NODEPS_REQUIREMENT_NAMES,
                ),
                _faiss_openblas_repair_command(ENV_NAME),
                "pip_install --force-reinstall -r /tmp/yubo-requirements-hyperscalees-final.txt",
                "pip_install --no-deps --force-reinstall -r /tmp/yubo-requirements-hyperscalees-final-nodeps.txt",
                (
                    "pip_install --index-url "
                    f"{shlex.quote(PYTORCH_CU128_INDEX_URL)} --extra-index-url https://pypi.nvidia.com "
                    "--no-deps --force-reinstall 'torchaudio==2.10.0'"
                ),
                f"mkdir -p {shlex.quote(HF_HOME_DIR)}",
                f"""run_in_env python - <<'PY'
from pathlib import Path
import importlib.util
import os

import faiss
import llvmlite
import modal
import numba
import numpy
import pkg_resources
import torch
import torchaudio
from brax import envs
from dm_control import suite
from numba import njit
from pyvecch.input_transforms import Identity


@njit
def _plus_one(x):
    return x + 1


assert _plus_one(1) == 2
assert numpy.__version__.startswith("2.2"), numpy.__version__
assert importlib.util.find_spec("vllm") is not None
envs.get_environment("ant")
env = suite.load("cartpole", "swingup")
env.close()
hf_home = Path({HF_HOME_DIR!r})
hf_home.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_home)
print(
    "MODAL_HYPERSCALEES_FINAL_OK",
    f"faiss={{faiss.__version__}}",
    f"identity={{Identity.__name__}}",
    f"numpy={{numpy.__version__}}",
    f"numba={{numba.__version__}}",
    f"llvmlite={{llvmlite.__version__}}",
    f"pkg_resources={{pkg_resources.__name__}}",
    f"torch={{torch.__version__}}",
    f"torchaudio={{torchaudio.__version__}}",
    f"modal={{getattr(modal, '__version__', 'unknown')}}",
)
PY""",
            ]
        )
    )


def _validate_isaaclab_runtime_command() -> str:
    return _bash(
        f"""
{_env_helpers(ISAACLAB_ENV_NAME)}
run_in_env python - <<'PY'
import importlib.util

import gymnasium
import imageio
import numpy
import torch
import torchaudio

assert importlib.util.find_spec("isaacsim") is not None
assert importlib.util.find_spec("isaaclab") is not None
assert importlib.util.find_spec("isaaclab_assets") is not None
assert importlib.util.find_spec("isaaclab_newton") is not None
assert importlib.util.find_spec("isaaclab_physx") is not None
assert importlib.util.find_spec("isaaclab_tasks") is not None
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

gymnasium.spec("Isaac-Velocity-Flat-G1-v0")
parse_env_cfg("Isaac-Velocity-Flat-G1-v0", num_envs=1, device="cpu")
print(
    "MODAL_ISAACLAB_RUNTIME_OK",
    f"gymnasium={{gymnasium.__version__}}",
    f"imageio={{imageio.__version__}}",
    f"numpy={{numpy.__version__}}",
    f"torch={{torch.__version__}}",
    f"torchaudio={{torchaudio.__version__}}",
)
PY
"""
    )


def _faiss_openblas_repair_command(env_name: str = ENV_NAME) -> str:
    specs = " ".join(shlex.quote(spec) for spec in FAISS_OPENBLAS_CONDA_SPECS)
    quoted_env = shlex.quote(env_name)
    return f"""
micromamba remove -y -n {quoted_env} faiss-cpu faiss libfaiss >/dev/null 2>&1 || true
run_in_env python -m pip uninstall -y faiss-cpu faiss >/dev/null 2>&1 || true
micromamba install -y -n {quoted_env} -c conda-forge {specs}
run_in_env python - <<'PY'
import faiss

print(f"FAISS_OPENBLAS_OK {{faiss.__version__}}")
PY
"""


def _env_helpers(env_name: str = ENV_NAME) -> str:
    quoted_env = shlex.quote(env_name)
    return f"""
ENV_PREFIX="$(micromamba run -n {quoted_env} python -c 'import sys; print(sys.prefix)')"
run_in_env() {{
  local env_ld_library_path="${{ENV_PREFIX}}/lib${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
  micromamba run -n {quoted_env} env "LD_LIBRARY_PATH=${{env_ld_library_path}}" "$@"
}}
pip_install() {{
  run_in_env python -m pip install --disable-pip-version-check "$@"
}}
"""


def _write_heredoc(path: str, content: str) -> str:
    return f"cat > {shlex.quote(path)} <<'EOF'\n{content.rstrip()}\nEOF"


def _read_repo_file(relpath: str) -> str:
    return (Path(__file__).resolve().parents[1] / relpath).read_text(encoding="utf-8")


def _select_requirements_command(source_path: str, output_path: str, package_names: tuple[str, ...]) -> str:
    pattern = "|".join(name.replace("-", "[-_]") for name in package_names)
    return "\n".join(
        [
            f"grep -E '^({pattern})(\\[|[<>=!~; ]|$)' {shlex.quote(source_path)} > {shlex.quote(output_path)}",
            f"test -s {shlex.quote(output_path)}",
        ]
    )


def _bash(command: str) -> str:
    payload = "set -euxo pipefail\n" + command.strip()
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return f'bash -lc "$(printf %s {shlex.quote(encoded)} | base64 -d)"'
