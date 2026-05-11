#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="yubo-rl"
PUFFERLIB_SPEC="pufferlib==3.0.0"
LASSOBENCH_RUNTIME_PACKAGES=(
  "ax-platform"
  "GPy>=1.9.2"
  "pyDOE>=0.3.8"
  "sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip"
)

usage() {
  cat <<'EOF'
Usage: bash admin/setup-rl.sh [options]

Options:
  --env-name <name>          Micromamba environment name (default: yubo-rl)
  -h, --help                 Show this help

Examples:
  bash admin/setup-rl.sh
  bash admin/setup-rl.sh --env-name my-env
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
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

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba not found in PATH." >&2
  exit 1
fi

if micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup-rl] env '${ENV_NAME}' already exists, reusing it"
else
  echo "[setup-rl] creating env '${ENV_NAME}' from admin/conda-rl.yml"
  micromamba env create -n "${ENV_NAME}" -f admin/conda-rl.yml
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

detect_torch_cuda_version() {
  local cuda_version
  cuda_version="$(python - <<'PY'
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
    echo "[setup-rl] failed to detect torch CUDA version" >&2
    exit 1
  fi

  echo "${cuda_version}"
}

echo "[setup-rl] installing requirements.txt"
python -m pip install -r requirements.txt

echo "[setup-rl] installing torchrl/tensordict pins"
python -m pip install torchrl==0.11.0 tensordict==0.11.0

echo "[setup-rl] installing BO extras (VecchiaBO, LassoBench, ennbo)"
python -m pip install celer
python -m pip install "${LASSOBENCH_RUNTIME_PACKAGES[@]}"
ENV_LIB="${CONDA_PREFIX}/lib" \
  LDFLAGS="-L${CONDA_PREFIX}/lib" \
  LIBRARY_PATH="${CONDA_PREFIX}/lib" \
  DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
  CPATH="$(python -c 'import pybind11; print(pybind11.get_include())')" \
  python -m pip install --no-build-isolation "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"
python -m pip install "LassoBench @ git+https://github.com/ksehic/LassoBench.git" --no-deps
python -m pip install ennbo --no-deps

OS_NAME="$(uname -s)"
if [[ "${OS_NAME}" == "Linux" ]]; then
  CUDA_VERSION="$(detect_torch_cuda_version)"
  echo "[setup-rl] linux detected, installing CUDA toolchain ${CUDA_VERSION} to match torch for pufferlib build"
  micromamba install -y -n "${ENV_NAME}" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_VERSION}" \
    "cuda-nvcc=${CUDA_VERSION}" \
    "cuda-cudart-dev=${CUDA_VERSION}" \
    ninja cmake gxx_linux-64

  echo "[setup-rl] installing ${PUFFERLIB_SPEC} --no-build-isolation --no-deps"
  export CUDA_HOME="${CONDA_PREFIX}"
  export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
  export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  python -m pip install --no-build-isolation --no-deps "${PUFFERLIB_SPEC}"
else
  echo "[setup-rl] macOS detected, skipping CUDA toolchain setup"
  echo "[setup-rl] installing ${PUFFERLIB_SPEC} --no-build-isolation --no-deps"
  python -m pip install --no-build-isolation --no-deps "${PUFFERLIB_SPEC}"
fi

echo "[setup-rl] validating pufferlib import"
python -c "from rl.pufferlib_compat import import_pufferlib_modules; import_pufferlib_modules(); print('pufferlib ok')"

echo "[setup-rl] done"
echo "activate with: micromamba activate ${ENV_NAME}"
