#!/usr/bin/env bash
set -euo pipefail

if ! command -v jj >/dev/null 2>&1; then
  echo "jj is not installed or is not on PATH." >&2
  exit 1
fi

if ! jj root >/dev/null 2>&1; then
  echo "No jj repository found." >&2
  echo "Initialize this checkout with jj first, then rerun this script." >&2
  exit 1
fi

include_sources='root-glob:"*.py" | root-glob:"**/*.py" | root-glob:"*.toml" | root-glob:"**/*.toml" | root-glob:"*.yaml" | root-glob:"**/*.yaml" | root-glob:"*.yml" | root-glob:"**/*.yml" | root-glob:"*.sh" | root-glob:"**/*.sh" | root-glob:"*.md" | root-glob:"**/*.md" | root-glob:"requirements*.txt" | root-glob:"**/requirements*.txt" | root-file:".gitattributes" | root-file:".gitignore" | root-file:".kissconfig" | root-file:".pre-commit-config.yaml" | root-file:"LICENSE" | root-file:"pixi.lock"'
exclude_artifacts='root:".malvin" | root:".pixi" | root:".pixi-state" | root:"_test" | root:"_test_results" | root:"smac3_output" | root:"data/mnist" | root-glob:".testmondata*" | root-glob:"**/__pycache__" | root-glob:"**/.pytest_cache" | root-glob:"**/.ruff_cache" | root-glob:"**/.mypy_cache" | root-glob:"**/.ipynb_checkpoints"'
auto_track="(${include_sources}) ~ (${exclude_artifacts})"

jj config set --repo snapshot.auto-track "'${auto_track}'"

echo "Configured jj snapshot.auto-track:"
jj config list --repo snapshot.auto-track
