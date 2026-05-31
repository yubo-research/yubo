#!/usr/bin/env bash
# Install repo-local jj aliases that delegate to admin/jj_check.sh.
set -euo pipefail

root="$(jj workspace root)"
script="${root}/admin/jj_check.sh"
if [[ ! -x "${script}" ]]; then
  chmod +x "${script}"
fi

legacy=(check certify evidence note exec doctor lab test-args)
for name in "${legacy[@]}"; do
  jj config unset --repo "aliases.${name}" 2>/dev/null || true
done

jj config set --repo aliases.check '["util", "exec", "--", "bash", "'"${script}"'", ""]'

echo "installed repo alias: jj check [quick|agent|publish]"
