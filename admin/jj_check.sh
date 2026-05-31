#!/usr/bin/env bash
# Gate runner for jj aliases (`jj check`, `jj check quick`, etc.).
set -euo pipefail

gate="agent"
for arg in "$@"; do
  if [[ -n "${arg}" ]]; then
    gate="${arg}"
    break
  fi
done

run_quick() {
  ruff check --config admin/ruff.toml --fix --fixable I001
  ruff format --config admin/ruff.toml
  ruff check --config admin/ruff.toml
  bash -lc "if rg -n '\\b(TEST|HACK)\\b' -g '*.py'; then exit 1; else exit 0; fi"
  admin/missing_files.sh
}

run_agent() {
  run_quick
  kiss check
  pixi run -e hyperscalees pytest -q
}

case "${gate}" in
  quick) run_quick ;;
  agent) run_agent ;;
  publish) run_agent ;;
  *)
    echo "unknown gate: ${gate} (expected quick, agent, or publish)" >&2
    exit 2
    ;;
esac
