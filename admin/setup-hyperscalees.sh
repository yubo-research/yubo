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
RUN_VERIFY=1
RECREATE_ENV=0
VLLM_STACK_PACKAGES=(
  "vllm==0.17.0"
  "ray==2.51.1"
  "peft==0.18.0"
  "transformers==4.57.1"
  "accelerate==1.11.0"
  "safetensors==0.6.2"
  "datasets==4.4.1"
  "pylatexenc==2.10"
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
MATH_VERIFY_SPEC="math-verify[antlr4_9_3]==0.9.0"
ANTLR_SPEC="antlr4-python3-runtime==4.9.3"

usage() {
  cat <<'EOF'
Usage: bash admin/setup-hyperscalees.sh [options]

Creates the CUDA-only EggRoll environment used for HyperscaleES-backed and
owned Qwen/vLLM LoRA Yubo experiments. This environment intentionally follows
the upstream ESHyperscale GPU stack instead of the yubo-rl dependency stack.

Options:
  --env-name <name>             Micromamba environment name (default: yubo-hyperscalees)
  --recreate-env                Remove and recreate the micromamba env first
  --repo-url <url>              HyperscaleES Git repository URL
  --commit <sha/ref>            HyperscaleES commit or ref to checkout
  --hf-home <path>              Hugging Face/JAX cache directory
  --jax-spec <spec>             JAX pip requirement (default: jax[cuda12]==0.8.1)
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
      ENV_NAME="$2"
      shift 2
      ;;
    --recreate-env)
      RECREATE_ENV=1
      shift
      ;;
    --repo-url)
      HYPERSCALEES_REPO_URL="$2"
      shift 2
      ;;
    --commit)
      HYPERSCALEES_REPO_COMMIT="$2"
      shift 2
      ;;
    --hf-home)
      HF_HOME_DIR="$2"
      shift 2
      ;;
    --jax-spec)
      JAX_SPEC="$2"
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
  echo "setup-hyperscalees.sh is CUDA/Linux-only. Use it on the remote GPU machine." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This EggRoll environment is intended only for CUDA machines." >&2
  exit 1
fi

if command -v micromamba >/dev/null 2>&1; then
  MICROMAMBA="$(command -v micromamba)"
elif [[ -x "${HOME}/.local/bin/micromamba" ]]; then
  MICROMAMBA="${HOME}/.local/bin/micromamba"
else
  echo "micromamba not found in PATH or at ${HOME}/.local/bin/micromamba." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git not found in PATH." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

env_exists() {
  "${MICROMAMBA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

if [[ "${RECREATE_ENV}" -eq 1 ]] && env_exists; then
  echo "[setup-hyperscalees] removing existing env '${ENV_NAME}'"
  "${MICROMAMBA}" env remove -y -n "${ENV_NAME}"
fi

if env_exists; then
  PY_MINOR="$("${MICROMAMBA}" run -n "${ENV_NAME}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${PY_MINOR}" != "${PYTHON_MINOR}" ]]; then
    echo "[setup-hyperscalees] env '${ENV_NAME}' uses Python ${PY_MINOR}, expected Python ${PYTHON_MINOR}." >&2
    echo "[setup-hyperscalees] Re-run with --recreate-env to rebuild the CUDA EggRoll env." >&2
    exit 1
  fi
  echo "[setup-hyperscalees] env '${ENV_NAME}' already exists, reusing it"
else
  echo "[setup-hyperscalees] creating env '${ENV_NAME}' from admin/conda-hyperscalees.yml"
  "${MICROMAMBA}" env create -y -n "${ENV_NAME}" -f admin/conda-hyperscalees.yml
fi

run_in_env() {
  "${MICROMAMBA}" run -n "${ENV_NAME}" "$@"
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
    echo "[setup-hyperscalees] optional module '${module}' already importable"
    return
  fi

  echo "[setup-hyperscalees] installing optional adapter package '${package}'"
  if ! run_in_env python -m pip install "${package}"; then
    if [[ "${package}" == "jaxmarl" ]]; then
      echo "[setup-hyperscalees] retrying optional package '${package}' without dependency resolution"
      if run_in_env python -m pip install --no-deps "${package}"; then
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
    echo "[setup-hyperscalees] optional module '${module}' already importable"
    return
  fi

  echo "[setup-hyperscalees] installing optional adapter package '${package}' without dependency resolution"
  if ! run_in_env python -m pip install --no-deps "${package}"; then
    echo "[setup-hyperscalees] warning: optional package '${package}' did not install; affected configs will report dependency_missing" >&2
  fi
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
  mkdir -p "${wheel_dir}"

  echo "[setup-hyperscalees] fetching HyperscaleES ${HYPERSCALEES_REPO_COMMIT}"
  git init -q "${source_dir}"
  git -C "${source_dir}" remote add origin "${HYPERSCALEES_REPO_URL}"
  git -C "${source_dir}" fetch -q --depth 1 origin "${HYPERSCALEES_REPO_COMMIT}"
  git -C "${source_dir}" checkout -q FETCH_HEAD

  echo "[setup-hyperscalees] repairing HyperscaleES package boundary"
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

  echo "[setup-hyperscalees] building HyperscaleES wheel"
  run_in_env python -m pip wheel --no-deps --wheel-dir "${wheel_dir}" "${source_dir}"
  wheel="$(find "${wheel_dir}" -maxdepth 1 -name 'hyperscalees-*.whl' -print -quit)"
  if [[ -z "${wheel}" ]]; then
    echo "[setup-hyperscalees] failed to build HyperscaleES wheel" >&2
    exit 1
  fi

  echo "[setup-hyperscalees] installing repaired HyperscaleES wheel: $(basename "${wheel}")"
  run_in_env python -m pip install --force-reinstall --no-deps "${wheel}"
  rm -rf "${build_root}"
}

echo "[setup-hyperscalees] upgrading pip"
run_in_env python -m pip install --upgrade pip

echo "[setup-hyperscalees] installing upstream-compatible CUDA JAX: ${JAX_SPEC}"
run_in_env python -m pip install --upgrade "${JAX_SPEC}"

echo "[setup-hyperscalees] installing Yubo runtime dependencies"
run_in_env python -m pip install click gymnasium datasets tyro tqdm wandb scipy

echo "[setup-hyperscalees] installing NumPy-2-compatible ENN/FAISS stack"
run_in_env python -m pip install \
  "ennbo==0.2.1" \
  "faiss-cpu==1.13.2" \
  "gpytorch==1.15.2" \
  "linear-operator==0.6.1" \
  "scikit-learn==1.8.0" \
  joblib \
  threadpoolctl

echo "[setup-hyperscalees] installing EggRoll JAX environment adapter dependencies"
run_in_env python -m pip install brax jumanji
install_optional_python_package craftax craftax
install_optional_python_package_no_deps jaxmarl jaxmarl
install_optional_python_package kinetix kinetix
install_optional_python_package navix navix

echo "[setup-hyperscalees] installing HyperscaleES runtime dependencies"
run_in_env python -m pip install "${HYPERSCALEES_RUNTIME_PACKAGES[@]}"

install_hyperscalees_package

echo "[setup-hyperscalees] installing math verifier compatible with HyperscaleES/Hydra"
run_in_env python -m pip install "${MATH_VERIFY_SPEC}" "${ANTLR_SPEC}"

echo "[setup-hyperscalees] installing owned Qwen/vLLM LoRA runtime dependencies"
run_in_env python -m pip install "${VLLM_STACK_PACKAGES[@]}"

echo "[setup-hyperscalees] re-asserting JAX and math-grader compatibility pins"
run_in_env python -m pip install --upgrade "${JAX_SPEC}" "${MATH_VERIFY_SPEC}" "${ANTLR_SPEC}"

mkdir -p "${HF_HOME_DIR}"
install_env_activation_hooks "${HF_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}"

if [[ "${RUN_VERIFY}" -eq 1 ]]; then
  echo "[setup-hyperscalees] checking ENN/FAISS NumPy ABI compatibility"
  run_in_env python - <<'PY'
import numpy
import faiss
import enn

assert int(numpy.__version__.split(".", 1)[0]) >= 2, numpy.__version__
print(f"[setup-hyperscalees] imports OK: numpy={numpy.__version__} faiss={faiss.__version__} enn={enn.__name__}")
PY
  echo "[setup-hyperscalees] checking HyperscaleES package metadata"
  run_in_env python - <<'PY'
import importlib.metadata

print(f"[setup-hyperscalees] hyperscalees package OK: {importlib.metadata.version('hyperscalees')}")
PY
  echo "[setup-hyperscalees] checking HyperscaleES LLM/pretrain imports"
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
  echo "[setup-hyperscalees] checking owned vLLM LoRA runtime imports"
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
fi

echo "[setup-hyperscalees] done"
echo "HyperscaleES and the owned Qwen/vLLM LoRA runtime are installed in '${ENV_NAME}'."
