#!/usr/bin/env bash
set -euo pipefail

artifact_re='(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.ipynb_checkpoints)/|^\.pixi/|^\.pixi-state/|^_test_results/|^smac3_output/|^\.testmondata$|\.py[cod]$|\.log$|\.tmp$'
source_re='(^|/)[^/]+\.(py|toml|ya?ml|sh|md)$|(^|/)requirements[^/]*\.txt$|^(\.gitattributes|\.gitignore|\.kissconfig|\.pre-commit-config\.yaml|LICENSE|pixi\.lock)$'

if jj root >/dev/null 2>&1; then
  # Snapshot first so jj's file list reflects the current working copy.
  status_output="$(jj --no-pager status)"

  tracked_files="$(jj --no-pager file list -r @)"
  tracked_artifacts="$(
    printf '%s\n' "${tracked_files}" | grep -E "${artifact_re}" || true
  )"

  if [[ -n "${tracked_artifacts}" ]]; then
    echo "Artifact files are tracked by jj:" >&2
    echo "${tracked_artifacts}" >&2
    echo "Remove them from tracking before committing." >&2
    exit 1
  fi

  untracked_source="$(
    printf '%s\n' "${status_output}" \
      | awk '/^\? / {print substr($0, 3)}' \
      | grep -E "${source_re}" \
      | grep -Ev "${artifact_re}" \
      || true
  )"

  if [[ -n "${untracked_source}" ]]; then
    echo "Source-like files are not tracked by jj:" >&2
    echo "${untracked_source}" >&2
    echo "Track intentional files with: jj file track <path>" >&2
    exit 1
  fi

  exit 0
fi

exec admin/missing_files.sh
