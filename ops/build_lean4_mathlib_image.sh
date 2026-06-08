#!/usr/bin/env bash
set -euo pipefail

runtime="${THM_DOCKER_RUNTIME:-docker}"
image="${THM_LEAN4_DOCKER_IMAGE:-yubo-lean4-mathlib:latest}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC2086
$runtime build \
  --build-arg "MATHLIB_BASE_IMAGE=${THM_LEAN4_BASE_IMAGE:-ghcr.io/leanprover-community/mathlib4/lean:latest}" \
  --build-arg "LEAN_TOOLCHAIN=${THM_LEAN4_TOOLCHAIN:-leanprover/lean4:v4.30.0-rc2}" \
  --build-arg "MATHLIB_REV=${THM_LEAN4_MATHLIB_REV:-v4.30.0-rc2}" \
  -t "$image" "$repo_root/docker/lean4-mathlib"
# shellcheck disable=SC2086
$runtime run --rm --network none --entrypoint /bin/sh "$image" -lc \
  'export HOME=/home/lean; export PATH=/home/lean/.elan/bin:$PATH; cd /workspace; printf "%s\n" "import Mathlib" "" "example : 1 + 1 = 2 := by norm_num" > /tmp/yubo_preflight.lean; yubo-lean /tmp/yubo_preflight.lean'
