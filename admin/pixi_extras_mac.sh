#!/usr/bin/env bash
# Idempotent Mac post-install: ennbo, LassoBench, menagerie; optional vLLM-metal (--llm).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "extras-mac is macOS-only; on Linux use: pixi run -e hyperscalees setup" >&2
  exit 1
fi

WITH_LLM=0
for arg in "$@"; do
  case "${arg}" in
    --llm) WITH_LLM=1 ;;
    -h | --help)
      echo "usage: pixi_extras_mac.sh [--llm]" >&2
      echo "  default: BO/JAX extras (ennbo, LassoBench, menagerie)" >&2
      echo "  --llm:   also build vLLM + vllm-metal from source (needs cargo)" >&2
      exit 0
      ;;
    *)
      echo "pixi_extras_mac.sh: unknown argument ${arg}" >&2
      exit 2
      ;;
  esac
done

ROOT="${PIXI_PROJECT_ROOT:-.}"
ENN_PATCH_ID="${ENN_PATCH_ID:-ennbo_0.3.10_delocate_v1}"
MARKER_BO="${CONDA_PREFIX}/.yubo_mac_extras_ready"
MARKER_PATCH="${CONDA_PREFIX}/.yubo_mac_extras_patch"
MARKER_LLM="${CONDA_PREFIX}/.yubo_mac_llm_ready"
PY="${CONDA_PREFIX}/bin/python"
PIP=( "${PY}" -m pip install --disable-pip-version-check )
export PATH="${CONDA_PREFIX}/bin:${PATH}"

_log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2
}

_run_with_heartbeat() {
  local label="$1"
  shift
  local interval="${YUBO_MAC_BUILD_HEARTBEAT_SECONDS:-30}"
  local start now elapsed pid status

  _log "start: ${label}"
  "$@" &
  pid="$!"
  start="$(date +%s)"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${interval}"
    if kill -0 "${pid}" 2>/dev/null; then
      now="$(date +%s)"
      elapsed=$((now - start))
      _log "still running: ${label} (${elapsed}s elapsed, pid ${pid})"
    fi
  done
  if wait "${pid}"; then
    status=0
  else
    status="$?"
  fi
  now="$(date +%s)"
  elapsed=$((now - start))
  if [[ "${status}" -ne 0 ]]; then
    _log "failed: ${label} (${elapsed}s, exit ${status})"
    return "${status}"
  fi
  _log "done: ${label} (${elapsed}s)"
}

_pin_runtime() {
  _run_with_heartbeat "pin runtime packages" \
    "${PIP[@]}" --force-reinstall --no-deps 'numpy>=2.3,<2.4' numba==0.62.1 llvmlite==0.45.1
}

_repair_faiss() {
  if ! "${PY}" -m pip show faiss-cpu >/dev/null 2>&1; then
    return 0
  fi
  faiss_cpu_ver="$("${PY}" -m pip show faiss-cpu | awk -F': ' '/^Version:/{print $2; exit}')"
  if [[ "${faiss_cpu_ver}" == "1.10.0" ]]; then
    return 0
  fi
  _log "removing pip faiss-cpu ${faiss_cpu_ver} (conda faiss 1.10 required)"
  "${PIP[@]}" uninstall -y faiss-cpu faiss 2>/dev/null || true
  if [[ -f "${ROOT}/pixi.toml" ]]; then
    (cd "${ROOT}" && pixi reinstall -e hyperscalees faiss faiss-cpu libfaiss)
  fi
}

_install_vllm_metal() {
  local machine
  machine="$("${PY}" -c "import platform; print(platform.machine())")"
  if [[ "${machine}" != "arm64" ]]; then
    echo "pixi_extras_mac.sh --llm: native arm64 Python required, got ${machine}" >&2
    exit 1
  fi

  if ! command -v cargo >/dev/null 2>&1; then
    echo "pixi_extras_mac.sh --llm: cargo not on PATH; install Rust from https://rustup.rs" >&2
    exit 1
  fi

  local vllm_v="0.21.0"
  local vllm_metal_ref="v0.2.0-20260528-103004"
  local start_dir tmp
  start_dir="$(pwd)"
  tmp="$(mktemp -d)"
  trap 'cd "${start_dir}" && rm -rf "${tmp}"' RETURN

  _log "vllm-metal Mac install into ${CONDA_PREFIX}"
  _log "temporary build directory: ${tmp}"

  _run_with_heartbeat "install MLX + HF stack" \
    "${PIP[@]}" \
    'mlx>=0.31.0,<0.32.0' \
    'mlx-lm>=0.31.3' \
    'mlx-vlm>=0.4.0,<0.5.0' \
    'transformers>=5.5.1' \
    'accelerate>=0.26.0' \
    'safetensors>=0.4.0' \
    'nanobind==2.10.2' \
    'maturin>=1.4,<2.0' \
    ninja

  _log "vLLM ${vllm_v}: download source tarball"
  cd "${tmp}"
  curl -fL --progress-bar -o "vllm-${vllm_v}.tar.gz" \
    "https://github.com/vllm-project/vllm/releases/download/v${vllm_v}/vllm-${vllm_v}.tar.gz"
  _log "vLLM ${vllm_v}: unpack source"
  tar xf "vllm-${vllm_v}.tar.gz"
  cd "vllm-${vllm_v}"
  _run_with_heartbeat "install vLLM CPU build requirements" \
    "${PIP[@]}" -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
  _run_with_heartbeat "build and install vLLM ${vllm_v} from source" \
    env CXXFLAGS="-Wno-parentheses" "${PIP[@]}" .
  cd "${tmp}"

  _log "vllm-metal ${vllm_metal_ref}: download source tarball"
  curl -fL --progress-bar -o vllm-metal.tar.gz \
    "https://github.com/vllm-project/vllm-metal/archive/refs/tags/${vllm_metal_ref}.tar.gz"
  _log "vllm-metal ${vllm_metal_ref}: unpack source"
  tar xf vllm-metal.tar.gz
  cd "vllm-metal-${vllm_metal_ref#v}"
  _run_with_heartbeat "build and install vllm-metal ${vllm_metal_ref}" \
    "${PIP[@]}" .

  _log "verify vLLM + vllm-metal imports"
  "${PY}" - <<'PY'
import vllm
import vllm_metal  # noqa: F401

print("vllm-metal ok", vllm.__version__)
PY

  cd "${start_dir}"
}

_run_bo_extras() {
  if [[ -f "${MARKER_BO}" ]] && [[ -f "${MARKER_PATCH}" ]] && [[ "$(cat "${MARKER_PATCH}")" == "${ENN_PATCH_ID}" ]]; then
    if "${PY}" -c 'import faiss; from enn.enn.enn_fitter import ENNStatefulFitter; from enn.enn.enn_class import EpistemicNearestNeighbors; import LassoBench' 2>/dev/null; then
      return 0
    fi
    _log "repairing Mac BO extras (stale marker)"
    rm -f "${MARKER_BO}" "${MARKER_PATCH}"
  elif [[ -f "${MARKER_BO}" ]]; then
    _log "rebuilding ennbo (${ENN_PATCH_ID})"
    rm -rf "${CONDA_PREFIX}/lib/python3."*/site-packages/enn" "${CONDA_PREFIX}/lib/python3."*/site-packages/enn-"*.dist-info 2>/dev/null || true
    rm -f "${MARKER_BO}" "${MARKER_PATCH}"
  fi
  _log "Mac BO/JAX extras (one-time)"

  bash "${ROOT}/admin/install_ennbo_mac.sh"

  if ! "${PY}" -c 'import LassoBench' 2>/dev/null; then
    (cd "${ROOT}" && pixi run -e hyperscalees install-lassobench)
  fi

  "${PY}" -c 'from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()' 2>/dev/null || true

  _pin_runtime
  _repair_faiss
  echo "${ENN_PATCH_ID}" > "${MARKER_PATCH}"
  touch "${MARKER_BO}"
}

_run_llm_extras() {
  if [[ -f "${MARKER_LLM}" ]]; then
    return 0
  fi
  if "${PY}" -c 'import vllm, vllm_metal' 2>/dev/null; then
    _log "Mac LLM imports already work; finishing post-install checks"
    _pin_runtime
    _repair_faiss
    touch "${MARKER_LLM}"
    return 0
  fi
  _log "Mac LLM extras (vllm-metal, one-time)"
  _install_vllm_metal
  _pin_runtime
  _repair_faiss
  touch "${MARKER_LLM}"
}

_run_bo_extras
if [[ "${WITH_LLM}" -eq 1 ]]; then
  _run_llm_extras
fi
