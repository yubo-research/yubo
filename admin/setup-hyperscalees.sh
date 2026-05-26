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
NVIDIA_PYPI_URL="https://pypi.nvidia.com"
ISAACSIM_SPEC="isaacsim[all,extscache]==6.0.0.0"
ISAACLAB_PIP_SPEC="isaaclab[isaacsim,all]"
ISAACLAB_SOURCE_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_SOURCE_REF="v3.0.0-beta"
ISAACLAB_SOURCE_DIR="${HOME}/.cache/yubo/isaaclab/IsaacLab"
ISAACLAB_SOURCE_INSTALL_TARGET="minimal"
LASSOBENCH_SPEC="LassoBench @ git+https://github.com/ksehic/LassoBench.git"
VECCHIABO_SPEC="git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"

RUN_VERIFY=1
RECREATE_ENV=0
ENV_WAS_CREATED=0

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
ENV_PREFIX="$("${MICROMAMBA}" run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"

run_in_env() {
  local env_ld_library_path="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  "${MICROMAMBA}" run -n "${ENV_NAME}" env "LD_LIBRARY_PATH=${env_ld_library_path}" "$@"
}

pip_install() {
  run_in_env python -m pip install --disable-pip-version-check "$@"
}

normalize_path_in_env() {
  run_in_env python - "$1" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
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
if not vocab.exists(): raise SystemExit(f"missing {vocab}")
init_text = init_file.read_text(encoding="utf-8")
version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
init_file.write_text(f'__version__ = "{version_match.group(1)}"\n', encoding="utf-8")
models_init.write_text("", encoding="utf-8")
pyproject_text = pyproject.read_text(encoding="utf-8")
old, new = 'jaxrwkv = ["tok_files/*"]', 'hyperscalees = ["tok_files/*"]'
pyproject.write_text(pyproject_text.replace(old, new), encoding="utf-8")
PY

  log "building HyperscaleES wheel"
  run_in_env python -m pip wheel --disable-pip-version-check --no-deps --wheel-dir "${wheel_dir}" "${source_dir}"
  wheel="$(find "${wheel_dir}" -maxdepth 1 -name 'hyperscalees-*.whl' -print -quit)"
  pip_install --no-deps "${wheel}"
}

install_vecchiabo_package() {
  if run_in_env python - <<'PY'
import faiss
from pyvecch.input_transforms import Identity

print(f"pyvecch/faiss imports OK: {faiss.__version__} {Identity.__name__}")
PY
  then
    log "pyvecch already importable"
    return
  fi
  log "refreshing conda faiss-cpu OpenBLAS runtime"
  "${MICROMAMBA}" remove -y -n "${ENV_NAME}" faiss-cpu faiss libfaiss >/dev/null 2>&1 || true
  run_in_env python -m pip uninstall -y faiss-cpu faiss >/dev/null 2>&1 || true
  "${MICROMAMBA}" install -y -n "${ENV_NAME}" -c conda-forge \
    nomkl \
    "libblas=*=*openblas" \
    openblas \
    "libfaiss=1.10.0=cpu_openblas*" \
    "faiss=1.10.0=cpu_openblas_py312*" \
    "faiss-cpu=1.10.0"
  run_in_env python - <<'PY'
import faiss

print(f"faiss import OK: {faiss.__version__}")
PY
  log "installing VecchiaBO / pyvecch"
  run_in_env bash -c '
    LDFLAGS="-L${CONDA_PREFIX}/lib" \
    LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    CPATH="$(python -c "import pybind11; print(pybind11.get_include())")" \
    python -m pip install --disable-pip-version-check --no-build-isolation --no-deps "'"${VECCHIABO_SPEC}"'"
  '
  run_in_env python - <<'PY'
import faiss
from pyvecch.input_transforms import Identity

print(f"pyvecch/faiss imports OK: {faiss.__version__} {Identity.__name__}")
PY
}

install_lassobench_package() {
  if run_in_env python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("LassoBench") else 1)'; then
    log "LassoBench already importable"
    return
  fi
  log "installing LassoBench"
  pip_install --no-deps "${LASSOBENCH_SPEC}"
}

install_isaaclab_stack() {
  if run_in_env python -c 'import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ("isaacsim", "isaaclab")) else 1)'; then
    log "Isaac Lab already importable"
    return
  fi
  log "installing Isaac Sim: ${ISAACSIM_SPEC}"
  pip_install --extra-index-url "${NVIDIA_PYPI_URL}" --pre "${ISAACSIM_SPEC}"
  log "trying Isaac Lab pip install: ${ISAACLAB_PIP_SPEC}"
  if pip_install --extra-index-url "${NVIDIA_PYPI_URL}" --pre "${ISAACLAB_PIP_SPEC}"; then return; fi
  log "falling back to Isaac Lab source ${ISAACLAB_SOURCE_REF}"
  mkdir -p "$(dirname "${ISAACLAB_SOURCE_DIR}")"
  if [[ -d "${ISAACLAB_SOURCE_DIR}/.git" ]]; then git -C "${ISAACLAB_SOURCE_DIR}" fetch --tags origin; else git clone "${ISAACLAB_SOURCE_URL}" "${ISAACLAB_SOURCE_DIR}"; fi
  git -C "${ISAACLAB_SOURCE_DIR}" checkout -q "${ISAACLAB_SOURCE_REF}"
  run_in_env bash -c "cd ${ISAACLAB_SOURCE_DIR} && ./isaaclab.sh --install ${ISAACLAB_SOURCE_INSTALL_TARGET}"
}

install_brax_compat_stack() {
  local specs=()
  local package spec
  for package in brax mujoco mujoco-mjx mujoco-warp warp-lang; do
    spec="$(grep -E "^${package}([<>=!~]|\\[|$)" admin/requirements-hyperscalees-nodeps.txt | head -n 1)"
    [[ -n "${spec}" ]] || die "missing ${package} pin in admin/requirements-hyperscalees-nodeps.txt"
    specs+=("${spec}")
  done
  log "ensuring Brax/MuJoCo compatibility pins: ${specs[*]}"
  pip_install --no-deps --force-reinstall "${specs[@]}"
  run_in_env python - <<'PY'
from brax import envs

envs.get_environment("ant")
print("BRAX_COMPAT_OK")
PY
}

install_repo_test_compat_stack() {
  log "ensuring repo test runtime pins"
  pip_install --force-reinstall \
    "setuptools>=77.0.3,<81.0.0" \
    "numpy>=2.2,<2.3" \
    "numba==0.61.2" \
    "llvmlite==0.44.0" \
    "dm-control>=1.0.40"
  run_in_env python - <<'PY'
import numpy
import numba
from dm_control import suite

assert numpy.__version__.startswith("2.2"), numpy.__version__
env = suite.load("cartpole", "swingup")
env.close()
print(f"REPO_TEST_COMPAT_OK numpy={numpy.__version__} numba={numba.__version__}")
PY
}

install_env_activation_hooks() {
  local hf_home="$1"
  run_in_env python - "${hf_home}" <<'PY'
import shlex, sys, pathlib
hf_home = pathlib.Path(sys.argv[1]).expanduser().resolve()
hf_home.mkdir(parents=True, exist_ok=True)
prefix = pathlib.Path(sys.prefix)
act = prefix / "etc" / "conda" / "activate.d" / "yubo-hyperscalees.sh"
deact = prefix / "etc" / "conda" / "deactivate.d" / "yubo-hyperscalees.sh"
act.parent.mkdir(parents=True, exist_ok=True)
deact.parent.mkdir(parents=True, exist_ok=True)
act.write_text(f"export HF_HOME={shlex.quote(hf_home.as_posix())}\nexport HF_HUB_CACHE={shlex.quote(hf_home.as_posix())}\nexport XLA_PYTHON_CLIENT_PREALLOCATE=false\n")
deact.write_text("unset HF_HOME\nunset HF_HUB_CACHE\nunset XLA_PYTHON_CLIENT_PREALLOCATE\n")
PY
}

HF_HOME_DIR="$(normalize_path_in_env "${HF_HOME_DIR}")"
ISAACLAB_SOURCE_DIR="$(normalize_path_in_env "${ISAACLAB_SOURCE_DIR}")"

if [[ "${ENV_WAS_CREATED}" -eq 1 ]]; then
  log "installing HyperscaleES environment requirements (including JAX: ${JAX_SPEC})"
  pip_install "${JAX_SPEC}" -r admin/requirements-hyperscalees.txt

  log "installing HyperscaleES environment no-deps requirements"
  pip_install --no-deps -r admin/requirements-hyperscalees-nodeps.txt

  install_hyperscalees_package
  install_vecchiabo_package
  install_lassobench_package
  install_isaaclab_stack
else
  log "env '${ENV_NAME}' already exists; performing incremental update"
  pip_install -r admin/requirements-hyperscalees.txt
  install_vecchiabo_package
  install_isaaclab_stack
fi
install_brax_compat_stack
install_repo_test_compat_stack

mkdir -p "${HF_HOME_DIR}"
install_env_activation_hooks "${HF_HOME_DIR}"

log "done"
echo "HyperscaleES stack is ready in '${ENV_NAME}'."
