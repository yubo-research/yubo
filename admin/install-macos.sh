#!/usr/bin/env bash
# DEPRECATED — use Pixi on Apple Silicon instead. See admin/README-mac-emps.md
set -euo pipefail
echo "admin/install-macos.sh is deprecated." >&2
echo "Use:" >&2
echo "  pixi install -e hyperscalees" >&2
echo "  pixi run -e hyperscalees bootstrap-mac   # or: extras-mac && check-mac" >&2
exit 1
