#!/usr/bin/env bash
set -euo pipefail

# Fail the commit if there are untracked "source-like" files.
# We intentionally ignore Python bytecode and cache directories.

untracked="$(
  git status --porcelain \
    | awk '$1 == "??" {print substr($0, 4)}' \
    | grep -vE '(^|/)(__pycache__/|\.pytest_cache/|\.mypy_cache/|\.ruff_cache/)' \
    | grep -vE '\.pyc$' \
    || true
)"

if [[ -n "${untracked}" ]]; then
  echo "Untracked files detected (please add to git or .gitignore):" >&2
  echo "${untracked}" >&2
  exit 1
fi

exit 0
