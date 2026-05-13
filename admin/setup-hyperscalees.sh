#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="yubo-hyperscalees"
PYTHON_MINOR="3.12"
HYPERSCALEES_REPO_URL="https://github.com/ESHyperscale/HyperscaleES.git"
HYPERSCALEES_REPO_COMMIT="b77f7d6f91238fd575313e946b9cad21e0a74b32"
HF_HOME_DIR="${HOME}/.cache/yubo/hyperscalees"
JAX_SPEC="jax[cuda12]==0.8.1"
JAX_NUMPY_COMPAT_SPEC="numpy==2.2.6"
SCIPY_COMPAT_SPEC="scipy==1.17.1"
PUFFERLIB_SPEC="pufferlib==3.0.0"
NVIDIA_PYPI_URL="https://pypi.nvidia.com"
ISAACSIM_SPEC="isaacsim[all,extscache]==6.0.0.0"
ISAACLAB_PIP_SPEC="isaaclab[isaacsim,all]"
ISAACLAB_SOURCE_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_SOURCE_REF="v3.0.0-beta"
ISAACLAB_SOURCE_DIR="${HOME}/.cache/yubo/isaaclab/IsaacLab"
ISAACLAB_SOURCE_INSTALL_TARGET="minimal"
LASSOBENCH_SPEC="LassoBench @ git+https://github.com/ksehic/LassoBench.git"
LASSOBENCH_RUNTIME_PACKAGES=(
  "ax-platform"
  "GPy>=1.9.2"
  "pyDOE>=0.3.8"
  "sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip"
)
RUN_VERIFY=1
RECREATE_ENV=0
ENV_WAS_CREATED=0
VLLM_STACK_PACKAGES=(
  "vllm==0.17.0"
  "ray==2.51.1"
  "peft==0.18.0"
  "transformers==4.57.1"
  "accelerate==1.11.0"
  "safetensors==0.6.2"
  "datasets==4.4.1"
  "pylatexenc==2.10"
  "nest-asyncio"
  "wandb==0.23.0"
  "weave==0.52.33"
)
HYPERSCALEES_RUNTIME_PACKAGES=(
  "huggingface_hub"
  "tokenizers"
  "einops"
  "importlib_resources"
  "pyrwkv-tokenizer"
  "hydra-core"
  "optax"
  "gymnax"
  "fire"
  "jupyter"
  "matplotlib"
  "reasoning-gym"
)
EGGROLL_JAX_RUNTIME_PACKAGES=(
  "absl-py==2.4.0"
  "chex==0.1.91"
  "dm-env==1.6"
  "esquilax==2.1.0"
  "etils==1.14.0"
  "flask==3.1.3"
  "flask-cors==6.0.2"
  "flax==0.12.7"
  "huggingface-hub==1.14.0"
  "jaxopt==0.8.5"
  "matplotlib==3.10.9"
  "ml-collections==1.1.0"
  "mujoco==3.8.0"
  "mujoco-mjx==3.8.0"
  "optax==0.2.8"
  "orbax-checkpoint==0.11.39"
  "pillow==12.2.0"
  "tensorboardx==2.6.5"
  "trimesh==4.12.2"
)
EGGROLL_JAX_TOPLEVEL_PACKAGES=(
  "brax==0.14.2"
  "jumanji==1.1.1"
)
EGGROLL_WARP_SUPPORT_PACKAGES=(
  "warp-lang==1.13.0"
  "mujoco-warp==3.8.0.3"
)
OPTIONAL_JAX_ENV_SUPPORT_PACKAGES=(
  "gymnax==0.0.9"
  "imageio==2.37.3"
  "networkx==3.6.1"
  "pandas==3.0.2"
  "pygame==2.6.1"
  "rlax==0.1.8"
  "seaborn==0.13.2"
)
YUBO_OPTIONAL_TEST_PACKAGES=(
  "botorch==0.12.0"
  "gpytorch==1.13"
  "linear-operator==0.5.3"
  "cma==4.0.0"
  "celer"
  "optuna==4.0.0"
  "smac==2.0.0"
  "modal==1.3.0.post1"
  "torchrl==0.11.0"
  "tensordict==0.11.0"
  "gymnasium[box2d]"
  "dm-control>=1.0.37"
  "pybind11==3.0.1"
)
YUBO_OPTIONAL_TEST_CONDA_PACKAGES=(
  "pyrfr"
  "swig"
)
VECCHIABO_SPEC="git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"
MATH_VERIFY_SPEC="math-verify[antlr4_9_3]==0.9.0"
ANTLR_SPEC="antlr4-python3-runtime==4.9.3"
PRIME_VERIFIERS_SPEC="verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git"
PRIME_VERIFIERS_GSM8K_SPEC="gsm8k @ git+https://github.com/PrimeIntellect-ai/verifiers.git#subdirectory=environments/gsm8k"

log() {
  echo "[setup-hyperscalees] $*"
}

die() {
  echo "[setup-hyperscalees] error: $*" >&2
  exit 1
}

need_value() {
  local opt="$1"
  local value="${2-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    die "${opt} requires a value"
  fi
}

usage() {
  cat <<'EOF'
Usage: bash admin/setup-hyperscalees.sh [options]

Creates the CUDA-only EggRoll environment used for HyperscaleES-backed and
owned Qwen/vLLM LoRA Yubo experiments. This environment intentionally follows
the upstream ESHyperscale GPU stack instead of the yubo-rl dependency stack.
If the env already exists, the script reuses it and only ensures the Isaac Lab
stack is present; use --recreate-env to rebuild the full stack from scratch.

Options:
  --env-name <name>             Micromamba environment name (default: yubo-hyperscalees)
  --recreate-env                Remove and recreate the micromamba env first
  --repo-url <url>              HyperscaleES Git repository URL
  --commit <sha/ref>            HyperscaleES commit or ref to checkout
  --hf-home <path>              Hugging Face/JAX cache directory
  --jax-spec <spec>             JAX pip requirement (default: jax[cuda12]==0.8.1)
  --isaaclab-ref <ref>          Isaac Lab source fallback ref (default: v3.0.0-beta)
  --isaaclab-source-dir <path>  Isaac Lab source fallback checkout path
  --isaaclab-install <target>   Isaac Lab source install target (default: minimal)
  --skip-verify                 Skip smoke checks after install
  -h, --help                    Show this help

Examples:
  bash admin/setup-hyperscalees.sh
  bash admin/setup-hyperscalees.sh --recreate-env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      need_value "$1" "${2-}"
      ENV_NAME="$2"
      shift 2
      ;;
    --recreate-env)
      RECREATE_ENV=1
      shift
      ;;
    --repo-url)
      need_value "$1" "${2-}"
      HYPERSCALEES_REPO_URL="$2"
      shift 2
      ;;
    --commit)
      need_value "$1" "${2-}"
      HYPERSCALEES_REPO_COMMIT="$2"
      shift 2
      ;;
    --hf-home)
      need_value "$1" "${2-}"
      HF_HOME_DIR="$2"
      shift 2
      ;;
    --jax-spec)
      need_value "$1" "${2-}"
      JAX_SPEC="$2"
      shift 2
      ;;
    --isaaclab-ref)
      need_value "$1" "${2-}"
      ISAACLAB_SOURCE_REF="$2"
      shift 2
      ;;
    --isaaclab-source-dir)
      need_value "$1" "${2-}"
      ISAACLAB_SOURCE_DIR="$2"
      shift 2
      ;;
    --isaaclab-install)
      need_value "$1" "${2-}"
      ISAACLAB_SOURCE_INSTALL_TARGET="$2"
      shift 2
      ;;
    --skip-verify)
      RUN_VERIFY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  die "setup-hyperscalees.sh is CUDA/Linux-only. Use it on the remote GPU machine."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. This EggRoll environment is intended only for CUDA machines."
fi

if command -v micromamba >/dev/null 2>&1; then
  MICROMAMBA="$(command -v micromamba)"
elif [[ -x "${HOME}/.local/bin/micromamba" ]]; then
  MICROMAMBA="${HOME}/.local/bin/micromamba"
else
  die "micromamba not found in PATH or at ${HOME}/.local/bin/micromamba."
fi

if ! command -v git >/dev/null 2>&1; then
  die "git not found in PATH."
fi

cd "${PROJECT_ROOT}"

env_exists() {
  "${MICROMAMBA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

if [[ "${RECREATE_ENV}" -eq 1 ]] && env_exists; then
  log "removing existing env '${ENV_NAME}'"
  "${MICROMAMBA}" env remove -y -n "${ENV_NAME}"
fi

if env_exists; then
  PY_MINOR="$("${MICROMAMBA}" run -n "${ENV_NAME}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${PY_MINOR}" != "${PYTHON_MINOR}" ]]; then
    die "env '${ENV_NAME}' uses Python ${PY_MINOR}, expected Python ${PYTHON_MINOR}. Re-run with --recreate-env."
  fi
  log "env '${ENV_NAME}' already exists, reusing it"
else
  log "creating env '${ENV_NAME}' from admin/conda-hyperscalees.yml"
  "${MICROMAMBA}" env create -y -n "${ENV_NAME}" -f admin/conda-hyperscalees.yml
  ENV_WAS_CREATED=1
fi

run_in_env() {
  "${MICROMAMBA}" run -n "${ENV_NAME}" "$@"
}

pip_install() {
  run_in_env python -m pip install --disable-pip-version-check "$@"
}

detect_torch_cuda_version() {
  local cuda_version
  cuda_version="$(run_in_env python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"failed to import torch while detecting CUDA version: {exc}")

cuda_version = torch.version.cuda
if not cuda_version:
    raise SystemExit("torch.version.cuda is empty; a CUDA-enabled torch build is required for pufferlib on Linux")

parts = cuda_version.split(".")
if len(parts) < 2:
    raise SystemExit(f"unexpected torch.version.cuda format: {cuda_version!r}")

print(".".join(parts[:2]))
PY
)"

  if [[ -z "${cuda_version}" ]]; then
    die "failed to detect torch CUDA version"
  fi

  echo "${cuda_version}"
}

normalize_path_in_env() {
  run_in_env python - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

install_optional_python_package() {
  local package="$1"
  local module="$2"
  if run_in_env python - "${module}" <<'PY'
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
  then
    log "optional module '${module}' already importable"
    return
  fi

  log "installing optional adapter package '${package}'"
  if ! pip_install "${package}"; then
    if [[ "${package}" == "jaxmarl" ]]; then
      log "retrying optional package '${package}' without dependency resolution"
      if pip_install --no-deps "${package}"; then
        return
      fi
    fi
    echo "[setup-hyperscalees] warning: optional package '${package}' did not install; affected configs will report dependency_missing" >&2
  fi
}

install_optional_python_package_no_deps() {
  local package="$1"
  local module="$2"
  if run_in_env python - "${module}" <<'PY'
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
  then
    log "optional module '${module}' already importable"
    return
  fi

  log "installing optional adapter package '${package}' without dependency resolution"
  if ! pip_install --no-deps "${package}"; then
    echo "[setup-hyperscalees] warning: optional package '${package}' did not install; affected configs will report dependency_missing" >&2
  fi
}

install_eggroll_jax_runtime_packages() {
  log "installing EggRoll JAX runtime support packages without touching the pinned JAX stack"
  pip_install "${EGGROLL_JAX_RUNTIME_PACKAGES[@]}" "${OPTIONAL_JAX_ENV_SUPPORT_PACKAGES[@]}"
  log "installing brax/jumanji without dependency resolution to preserve pinned JAX"
  pip_install --no-deps "${EGGROLL_JAX_TOPLEVEL_PACKAGES[@]}"
  log "installing warp/mujoco-warp without dependency resolution to preserve pinned JAX"
  pip_install --no-deps "${EGGROLL_WARP_SUPPORT_PACKAGES[@]}"
}

check_glibc_for_isaaclab() {
  local glibc_version
  if ! command -v ldd >/dev/null 2>&1; then
    die "ldd not found; cannot verify glibc compatibility for Isaac Sim."
  fi

  glibc_version="$(ldd --version | head -n 1 | grep -Eo '[0-9]+\.[0-9]+' | head -n 1 || true)"
  if [[ -z "${glibc_version}" ]]; then
    die "failed to detect glibc version for Isaac Sim."
  fi
  if ! python3 - "${glibc_version}" <<'PY'
import sys

parts = tuple(int(p) for p in sys.argv[1].split(".")[:2])
raise SystemExit(0 if parts >= (2, 35) else 1)
PY
  then
    die "glibc ${glibc_version} is too old. Isaac Sim pip installs require glibc >= 2.35."
  fi
}

warn_if_isaac_torch_mismatch() {
  run_in_env python - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"[setup-hyperscalees] warning: torch import failed before Isaac install: {exc}")
    raise SystemExit(0)

version = getattr(torch, "__version__", "unknown")
if not version.startswith("2.10.0"):
    print(
        "[setup-hyperscalees] warning: Isaac Lab 3.0 docs recommend "
        f"torch==2.10.0; Isaac Lab's source installer may reconcile torch from: {version}"
    )
else:
    print(f"[setup-hyperscalees] Isaac Lab torch prerequisite already matches: {version}")
PY
}

install_isaaclab_stack() {
  check_glibc_for_isaaclab
  warn_if_isaac_torch_mismatch

  if run_in_env python - <<'PY'
import importlib.util

mods = ("isaacsim", "isaaclab", "isaaclab_newton")
raise SystemExit(0 if all(importlib.util.find_spec(name) is not None for name in mods) else 1)
PY
  then
    log "Isaac Sim / Isaac Lab / Newton already importable"
    return
  fi

  log "installing Isaac Sim into '${ENV_NAME}': ${ISAACSIM_SPEC}"
  pip_install --extra-index-url "${NVIDIA_PYPI_URL}" --pre "${ISAACSIM_SPEC}"

  log "trying Isaac Lab Python package in '${ENV_NAME}': ${ISAACLAB_PIP_SPEC}"
  if pip_install --extra-index-url "${NVIDIA_PYPI_URL}" --pre "${ISAACLAB_PIP_SPEC}"; then
    return
  fi

  log "Isaac Lab wheel was not available for this Python; falling back to source ${ISAACLAB_SOURCE_REF}"
  mkdir -p "$(dirname "${ISAACLAB_SOURCE_DIR}")"
  if [[ -d "${ISAACLAB_SOURCE_DIR}/.git" ]]; then
    log "updating Isaac Lab source checkout at ${ISAACLAB_SOURCE_DIR}"
    git -C "${ISAACLAB_SOURCE_DIR}" fetch --tags origin
  else
    log "cloning Isaac Lab source into ${ISAACLAB_SOURCE_DIR}"
    git clone "${ISAACLAB_SOURCE_URL}" "${ISAACLAB_SOURCE_DIR}"
  fi

  git -C "${ISAACLAB_SOURCE_DIR}" checkout -q "${ISAACLAB_SOURCE_REF}"
  git -C "${ISAACLAB_SOURCE_DIR}" submodule update --init --recursive

  log "installing Isaac Lab source target '${ISAACLAB_SOURCE_INSTALL_TARGET}' into '${ENV_NAME}'"
  if [[ "${ISAACLAB_SOURCE_INSTALL_TARGET}" == "all" ]]; then
    log "note: 'all' pulls mimic/robomimic stacks that may require extra system build deps (e.g. egl-probe/cmake); prefer 'minimal' unless you need the full suite"
  fi
  run_in_env bash -lc '
    set -euo pipefail
    cd "$1"
    ./isaaclab.sh --install "$2"
  ' bash "${ISAACLAB_SOURCE_DIR}" "${ISAACLAB_SOURCE_INSTALL_TARGET}"
}

verify_isaaclab_stack() {
  log "checking Isaac Lab / Newton imports"
  run_in_env python - <<'PY'
import importlib
import importlib.util
import torch

isaacsim_spec = importlib.util.find_spec("isaacsim")
if isaacsim_spec is None:
    raise SystemExit("isaacsim package is not installed")
print(f"[setup-hyperscalees] isaacsim package OK: {isaacsim_spec.origin}")

mods = [
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_newton",
    "warp",
    "mujoco",
    "mujoco_warp",
]
for name in mods:
    mod = importlib.import_module(name)
    print(f"[setup-hyperscalees] import OK: {name} -> {getattr(mod, '__file__', 'unknown')}")
print(f"[setup-hyperscalees] torch import OK for Isaac stack: {torch.__version__}")
PY
}

install_env_activation_hooks() {
  local hf_home="$1"
  run_in_env python - "${hf_home}" <<'PY'
import shlex
import sys
from pathlib import Path

hf_home = Path(sys.argv[1]).expanduser().resolve()
hf_home.mkdir(parents=True, exist_ok=True)

prefix = Path(sys.prefix)
activate_dir = prefix / "etc" / "conda" / "activate.d"
deactivate_dir = prefix / "etc" / "conda" / "deactivate.d"
activate_dir.mkdir(parents=True, exist_ok=True)
deactivate_dir.mkdir(parents=True, exist_ok=True)

activate_file = activate_dir / "yubo-hyperscalees.sh"
deactivate_file = deactivate_dir / "yubo-hyperscalees.sh"
quoted_hf_home = shlex.quote(hf_home.as_posix())

activate_file.write_text(
    "#!/usr/bin/env bash\n"
    "export _YUBO_HYPERSCALEES_OLD_HF_HOME=\"${HF_HOME-}\"\n"
    "export _YUBO_HYPERSCALEES_OLD_HF_HUB_CACHE=\"${HF_HUB_CACHE-}\"\n"
    "export _YUBO_HYPERSCALEES_OLD_XLA_PYTHON_CLIENT_PREALLOCATE=\"${XLA_PYTHON_CLIENT_PREALLOCATE-}\"\n"
    f"export HF_HOME={quoted_hf_home}\n"
    f"export HF_HUB_CACHE={quoted_hf_home}\n"
    "export XLA_PYTHON_CLIENT_PREALLOCATE=false\n",
    encoding="utf-8",
)
deactivate_file.write_text(
    "#!/usr/bin/env bash\n"
    "if [[ -n \"${_YUBO_HYPERSCALEES_OLD_HF_HOME+x}\" ]]; then\n"
    "  if [[ -n \"${_YUBO_HYPERSCALEES_OLD_HF_HOME}\" ]]; then\n"
    "    export HF_HOME=\"${_YUBO_HYPERSCALEES_OLD_HF_HOME}\"\n"
    "  else\n"
    "    unset HF_HOME\n"
    "  fi\n"
    "  unset _YUBO_HYPERSCALEES_OLD_HF_HOME\n"
    "fi\n"
    "if [[ -n \"${_YUBO_HYPERSCALEES_OLD_HF_HUB_CACHE+x}\" ]]; then\n"
    "  if [[ -n \"${_YUBO_HYPERSCALEES_OLD_HF_HUB_CACHE}\" ]]; then\n"
    "    export HF_HUB_CACHE=\"${_YUBO_HYPERSCALEES_OLD_HF_HUB_CACHE}\"\n"
    "  else\n"
    "    unset HF_HUB_CACHE\n"
    "  fi\n"
    "  unset _YUBO_HYPERSCALEES_OLD_HF_HUB_CACHE\n"
    "fi\n"
    "if [[ -n \"${_YUBO_HYPERSCALEES_OLD_XLA_PYTHON_CLIENT_PREALLOCATE+x}\" ]]; then\n"
    "  if [[ -n \"${_YUBO_HYPERSCALEES_OLD_XLA_PYTHON_CLIENT_PREALLOCATE}\" ]]; then\n"
    "    export XLA_PYTHON_CLIENT_PREALLOCATE=\"${_YUBO_HYPERSCALEES_OLD_XLA_PYTHON_CLIENT_PREALLOCATE}\"\n"
    "  else\n"
    "    unset XLA_PYTHON_CLIENT_PREALLOCATE\n"
    "  fi\n"
    "  unset _YUBO_HYPERSCALEES_OLD_XLA_PYTHON_CLIENT_PREALLOCATE\n"
    "fi\n",
    encoding="utf-8",
)
print(f"[setup-hyperscalees] installing activation hooks for HF cache at {hf_home}")
PY
}

install_hyperscalees_package() {
  local build_root source_dir wheel_dir wheel
  build_root="$(mktemp -d)"
  source_dir="${build_root}/HyperscaleES"
  wheel_dir="${build_root}/wheelhouse"
  cleanup_hyperscalees_build() {
    rm -rf "${build_root}"
    trap - RETURN
  }
  trap cleanup_hyperscalees_build RETURN
  mkdir -p "${wheel_dir}"

  log "fetching HyperscaleES ${HYPERSCALEES_REPO_COMMIT}"
  git init -q "${source_dir}"
  git -C "${source_dir}" remote add origin "${HYPERSCALEES_REPO_URL}"
  git -C "${source_dir}" fetch -q --depth 1 origin "${HYPERSCALEES_REPO_COMMIT}"
  git -C "${source_dir}" checkout -q FETCH_HEAD

  log "repairing HyperscaleES package boundary"
  run_in_env python - "${source_dir}" <<'PY'
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
    raise SystemExit(f"missing expected HyperscaleES tokenizer asset: {vocab}")

init_text = init_file.read_text(encoding="utf-8")
version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
if version_match is None:
    raise SystemExit("could not find hyperscalees.__version__")
init_file.write_text(f'__version__ = "{version_match.group(1)}"\n', encoding="utf-8")

models_init.write_text("", encoding="utf-8")

pyproject_text = pyproject.read_text(encoding="utf-8")
old = 'jaxrwkv = ["tok_files/*"]'
new = 'hyperscalees = ["tok_files/*"]'
if old not in pyproject_text:
    raise SystemExit(f"could not find package-data entry {old!r}")
pyproject.write_text(pyproject_text.replace(old, new), encoding="utf-8")
PY

  log "building HyperscaleES wheel"
  run_in_env python -m pip wheel --disable-pip-version-check --no-deps --wheel-dir "${wheel_dir}" "${source_dir}"
  wheel="$(find "${wheel_dir}" -maxdepth 1 -name 'hyperscalees-*.whl' -print -quit)"
  if [[ -z "${wheel}" ]]; then
    echo "[setup-hyperscalees] error: failed to build HyperscaleES wheel" >&2
    return 1
  fi

  log "installing repaired HyperscaleES wheel: $(basename "${wheel}")"
  pip_install --no-deps "${wheel}"
}

install_vecchiabo_package() {
  if run_in_env python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pyvecch") is not None else 1)
PY
  then
    log "optional module 'pyvecch' already importable"
    return
  fi

  log "installing VecchiaBO / pyvecch"
  run_in_env bash -lc '
    ENV_LIB="${CONDA_PREFIX}/lib" \
    LDFLAGS="-L${CONDA_PREFIX}/lib" \
    LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
    CPATH="$(python -c '"'"'import pybind11; print(pybind11.get_include())'"'"')" \
    python -m pip install --disable-pip-version-check --no-build-isolation "'"${VECCHIABO_SPEC}"'"
  '
}

install_lassobench_package() {
  log "installing LassoBench runtime dependencies"
  pip_install "${LASSOBENCH_RUNTIME_PACKAGES[@]}"

  if run_in_env python - <<'PY'
import LassoBench  # noqa: F401
PY
  then
    log "optional module 'LassoBench' already importable"
    return
  fi

  log "installing LassoBench"
  pip_install --no-deps "${LASSOBENCH_SPEC}"
}

install_pufferlib_package() {
  local cuda_version
  if run_in_env python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pufferlib") is not None else 1)
PY
  then
    log "optional module 'pufferlib' already importable"
    return
  fi

  cuda_version="$(detect_torch_cuda_version)"
  log "installing CUDA toolchain ${cuda_version} to match torch for pufferlib build"
  "${MICROMAMBA}" install -y -n "${ENV_NAME}" -c nvidia -c conda-forge \
    "cuda-toolkit=${cuda_version}" \
    "cuda-nvcc=${cuda_version}" \
    "cuda-cudart-dev=${cuda_version}" \
    ninja cmake gxx_linux-64

  log "installing ${PUFFERLIB_SPEC} without build isolation"
  run_in_env bash -lc '
    export CUDA_HOME="${CONDA_PREFIX}"
    export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
    export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
    export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    python -m pip install --disable-pip-version-check --no-build-isolation --no-deps "'"${PUFFERLIB_SPEC}"'"
  '
}

install_optional_test_conda_packages() {
  log "installing conda-side BO/test extras"
  "${MICROMAMBA}" install -y -n "${ENV_NAME}" -c conda-forge "${YUBO_OPTIONAL_TEST_CONDA_PACKAGES[@]}"
}

HF_HOME_DIR="$(normalize_path_in_env "${HF_HOME_DIR}")"
ISAACLAB_SOURCE_DIR="$(normalize_path_in_env "${ISAACLAB_SOURCE_DIR}")"

if [[ "${ENV_WAS_CREATED}" -eq 1 ]]; then
  log "installing upstream-compatible CUDA JAX: ${JAX_SPEC}"
  pip_install "${JAX_SPEC}" "${JAX_NUMPY_COMPAT_SPEC}" "${SCIPY_COMPAT_SPEC}"

  log "installing Yubo runtime dependencies"
  pip_install click gymnasium tyro tqdm

  log "installing NumPy-2-compatible ENN/FAISS stack"
  pip_install \
    --no-deps "ennbo==0.2.1" \
    "faiss-cpu==1.13.2" \
    "scikit-learn==1.8.0" \
    joblib \
    threadpoolctl

  install_eggroll_jax_runtime_packages
  install_optional_python_package_no_deps craftax craftax
  install_optional_python_package_no_deps jaxmarl jaxmarl
  install_optional_python_package_no_deps kinetix kinetix
  install_optional_python_package_no_deps navix navix

  log "installing HyperscaleES runtime dependencies"
  pip_install "${HYPERSCALEES_RUNTIME_PACKAGES[@]}"

  install_hyperscalees_package

  log "installing math verifier compatible with HyperscaleES/Hydra"
  pip_install "${MATH_VERIFY_SPEC}" "${ANTLR_SPEC}"

  log "installing Prime Intellect verifiers library and GSM8K environment"
  pip_install "${PRIME_VERIFIERS_SPEC}" "${PRIME_VERIFIERS_GSM8K_SPEC}"

  log "installing owned Qwen/vLLM LoRA runtime dependencies"
  pip_install "${VLLM_STACK_PACKAGES[@]}"

  log "installing Yubo BO/RL extras used by the broader repo suite"
  install_optional_test_conda_packages
  pip_install "${YUBO_OPTIONAL_TEST_PACKAGES[@]}"
  install_vecchiabo_package
  install_lassobench_package
  install_pufferlib_package
  install_isaaclab_stack
else
  log "env '${ENV_NAME}' already exists; installing incremental verifier dependency and Isaac Lab stack"
  log "use --recreate-env to rebuild the full environment from scratch with the current package set"
  pip_install "${PRIME_VERIFIERS_SPEC}" "${PRIME_VERIFIERS_GSM8K_SPEC}"
  install_isaaclab_stack
fi

mkdir -p "${HF_HOME_DIR}"
install_env_activation_hooks "${HF_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}"

if [[ "${RUN_VERIFY}" -eq 1 ]]; then
  log "checking ENN/FAISS NumPy ABI compatibility"
  run_in_env python - <<'PY'
import numpy
import faiss
import enn

assert int(numpy.__version__.split(".", 1)[0]) >= 2, numpy.__version__
print(f"[setup-hyperscalees] imports OK: numpy={numpy.__version__} faiss={faiss.__version__} enn={enn.__name__}")
PY
  log "checking HyperscaleES package metadata"
  run_in_env python - <<'PY'
import importlib.metadata

print(f"[setup-hyperscalees] hyperscalees package OK: {importlib.metadata.version('hyperscalees')}")
PY
  log "checking HyperscaleES LLM/pretrain imports"
  run_in_env python - <<'PY'
from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.llm.auto import get_model
from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
from hyperscalees.noiser.base_noiser import Noiser

tok = LegacyWorldTokenizer()
assert len(tok.encode("1 + 1")) > 0
assert "gsm8k" in all_tasks
assert "gsm8k" in validation_tasks
print(
    "[setup-hyperscalees] HyperscaleES LLM imports OK: "
    f"tasks={len(all_tasks)} tokenizer={type(tok).__name__} "
    f"model_loader={get_model.__name__} noiser={Noiser.__name__} tree_key={simple_es_tree_key.__name__}"
)
PY
  log "checking owned vLLM LoRA runtime imports"
  run_in_env python - <<'PY'
import accelerate
import importlib.metadata
import jax
import math_verify
import peft
import ray
import safetensors
import scipy
import transformers
import vllm

antlr_version = importlib.metadata.version("antlr4-python3-runtime")
assert antlr_version.startswith("4.9."), antlr_version

print(f"[setup-hyperscalees] vllm import OK: {vllm.__version__}")
print(f"[setup-hyperscalees] ray import OK: {ray.__version__}")
print(f"[setup-hyperscalees] transformers import OK: {transformers.__version__}")
print(f"[setup-hyperscalees] peft import OK: {peft.__version__}")
print(f"[setup-hyperscalees] accelerate import OK: {accelerate.__version__}")
print(f"[setup-hyperscalees] safetensors import OK: {safetensors.__version__}")
print(f"[setup-hyperscalees] math_verify import OK: {math_verify.__name__}")
print(f"[setup-hyperscalees] antlr runtime OK: {antlr_version}")
print(f"[setup-hyperscalees] scipy import OK: {scipy.__version__}")
print(f"[setup-hyperscalees] jax import OK after vLLM install: {jax.__version__}")
PY
  log "checking Prime Intellect verifiers import"
  run_in_env python - <<'PY'
import importlib.metadata
import gsm8k
import verifiers

print(f"[setup-hyperscalees] verifiers import OK: {importlib.metadata.version('verifiers')}")
print(f"[setup-hyperscalees] verifiers gsm8k environment import OK: {gsm8k.__name__}")
PY
  log "checking BO/RL optional extras expected by repo tests"
  run_in_env python - <<'PY'
import cma
import importlib
from rl.pufferlib_compat import import_pufferlib_modules

import_pufferlib_modules()
smac = importlib.import_module("smac")
pyrfr = importlib.import_module("pyrfr")
lasso = importlib.import_module("LassoBench")
pyvecch = importlib.import_module("pyvecch")
warp = importlib.import_module("warp")
mujoco_warp = importlib.import_module("mujoco_warp")

print(f"[setup-hyperscalees] cma import OK: {cma.__version__}")
print(f"[setup-hyperscalees] smac import OK: {getattr(smac, '__version__', 'unknown')}")
print(f"[setup-hyperscalees] pyrfr import OK: {getattr(pyrfr, '__file__', 'unknown')}")
print(f"[setup-hyperscalees] LassoBench import OK: {getattr(lasso, '__file__', 'unknown')}")
print(f"[setup-hyperscalees] pyvecch import OK: {getattr(pyvecch, '__file__', 'unknown')}")
print(f"[setup-hyperscalees] warp import OK: {getattr(warp, '__version__', 'unknown')}")
print(f"[setup-hyperscalees] mujoco_warp import OK: {getattr(mujoco_warp, '__version__', 'unknown')}")
print("[setup-hyperscalees] pufferlib import OK")
PY
  verify_isaaclab_stack
fi

log "done"
echo "HyperscaleES, the owned Qwen/vLLM LoRA runtime, the repo BO/RL extras, and Isaac Sim / Isaac Lab / Newton are ready in '${ENV_NAME}'."
