#!/usr/bin/env bash
set -euo pipefail

# Fail the commit if there are untracked Python source files.
# We intentionally ignore Python bytecode and cache directories.

git_untracked_py="$(
  git status --porcelain \
    | awk '$1 == "??" {print substr($0, 4)}' \
    | grep -vE '(^|/)(__pycache__/|\.pytest_cache/|\.mypy_cache/|\.ruff_cache/)' \
    | grep -E '\.py$' \
    || true
)"

jj_available=0
if command -v jj >/dev/null 2>&1 && jj root >/dev/null 2>&1; then
  jj_available=1
fi

jj_tracks_file() {
  local path="$1"
  jj --ignore-working-copy file list -r @ "${path}" 2>/dev/null | grep -Fxq "${path}"
}

untracked_py=""
while IFS= read -r path; do
  [[ -z "${path}" ]] && continue
  if [[ "${jj_available}" == "1" ]] && jj_tracks_file "${path}"; then
    continue
  fi
  untracked_py+="${path}"$'\n'
done <<< "${git_untracked_py}"
untracked_py="${untracked_py%$'\n'}"

if [[ -n "${untracked_py}" ]]; then
  echo "Untracked Python files detected (please add to git or .gitignore):" >&2
  echo "${untracked_py}" >&2
  exit 1
fi

exit 0
