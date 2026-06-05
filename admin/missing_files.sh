#!/usr/bin/env bash
set -euo pipefail

# Fail the commit if there are untracked Python source files.
# We intentionally ignore Python bytecode and cache directories.

untracked_py="$(
  git status --porcelain \
    | awk '$1 == "??" {print substr($0, 4)}' \
    | grep -vE '(^|/)(__pycache__/|\.pytest_cache/|\.mypy_cache/|\.ruff_cache/)' \
    | grep -E '\.py$' \
    || true
)"

if [[ -n "${untracked_py}" ]]; then
  echo "Untracked Python files detected (please add to git or .gitignore):" >&2
  echo "${untracked_py}" >&2
  exit 1
fi

exit 0
