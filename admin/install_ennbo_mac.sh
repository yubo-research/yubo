#!/usr/bin/env bash
# Install upstream ennbo on macOS: conda faiss at runtime, delocated wheel for enn_rust.
# PyPI 0.3.10 is wheel-only (cp311 mac / cp312 linux); py313 mac builds from git @ ENNBO_REF.
# Build follows yubo-research/enn scripts/build_wheels.sh (Homebrew faiss at build time only).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "install_ennbo_mac.sh is macOS-only" >&2
  exit 1
fi

ROOT="${PIXI_PROJECT_ROOT:-.}"
ENNBO_VERSION="${ENNBO_VERSION:-0.3.10}"
ENNBO_GIT="${ENNBO_GIT:-https://github.com/yubo-research/enn.git}"
ENNBO_REF="${ENNBO_REF:-baf6312fc25384bf320dff9757b7e92ce375bf6c}"
PY="${CONDA_PREFIX:?CONDA_PREFIX must be set}/bin/python"
PIP=( "${PY}" -m pip install --disable-pip-version-check )

_ennbo_ok() {
  "${PY}" -c "
import importlib.metadata as m
import faiss
from enn.enn.enn_class import EpistemicNearestNeighbors
assert m.version('ennbo') == '${ENNBO_VERSION}', m.version('ennbo')
print('ennbo', m.version('ennbo'), 'faiss', faiss.__version__, EpistemicNearestNeighbors.__name__)
"
}

if _ennbo_ok 2>/dev/null; then
  exit 0
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "install_ennbo_mac.sh: Homebrew required to link enn_rust (build-time only)" >&2
  exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "install_ennbo_mac.sh: cargo not on PATH (pixi rust or rustup)" >&2
  exit 1
fi

echo "=== ennbo ${ENNBO_VERSION} @ ${ENNBO_REF} (delocate wheel into ${CONDA_PREFIX}) ==="
brew list faiss >/dev/null 2>&1 || brew install faiss
brew list libomp >/dev/null 2>&1 || brew install libomp

"${PIP[@]}" -q maturin 'delocate>=0.13'

tmp="$(mktemp -d)"
# shellcheck disable=SC2064
trap "rm -rf '${tmp}'" RETURN

git clone --filter=blob:none --depth 1 "${ENNBO_GIT}" "${tmp}/enn"
if ! git -C "${tmp}/enn" checkout -q "${ENNBO_REF}" 2>/dev/null; then
  git -C "${tmp}/enn" fetch --depth 1 origin "${ENNBO_REF}"
  git -C "${tmp}/enn" checkout -q FETCH_HEAD
fi
src="${tmp}/enn"
pkg_ver="$(grep -E '^version = ' "${src}/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')"
if [[ "${pkg_ver}" != "${ENNBO_VERSION}" ]]; then
  echo "install_ennbo_mac.sh: expected ennbo ${ENNBO_VERSION}, git has ${pkg_ver}" >&2
  exit 1
fi

if grep -q 'auditwheel = "repair"' "${src}/pyproject.toml"; then
  sed -i '' 's/auditwheel = "repair"/auditwheel = "skip"/' "${src}/pyproject.toml"
fi
"${PY}" "${ROOT}/admin/patch_enn_failure_tolerance_dim.py" "${src}"

py_tag="$("${PY}" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
# Match PyPI mac wheel (macosx_26_0_arm64) and Homebrew faiss lib min version.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"
export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"

rm -rf "${src}/dist"
mkdir -p "${src}/dist/raw" "${src}/dist/repaired"
(
  cd "${src}"
  "${PY}" -m maturin build --release --interpreter "${PY}" --auditwheel skip --out dist/raw
  delocate-wheel -w dist/repaired -v dist/raw/*"${py_tag}"*.whl
)

"${PIP[@]}" --no-deps --force-reinstall "${src}/dist/repaired"/*"${py_tag}"*.whl
_ennbo_ok
