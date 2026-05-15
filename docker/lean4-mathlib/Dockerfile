ARG MATHLIB_BASE_IMAGE=ghcr.io/leanprover-community/mathlib4/lean:latest
FROM ${MATHLIB_BASE_IMAGE}

ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.30.0-rc2
ARG MATHLIB_REV=v4.30.0-rc2

USER root
RUN mkdir -p /workspace/Yubo && chown -R lean:lean /workspace

USER lean
WORKDIR /workspace

RUN cat > lakefile.lean <<EOF
import Lake
open Lake DSL

package yubo_thm where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "${MATHLIB_REV}"

@[default_target]
lean_lib Yubo where
EOF

RUN cat > lean-toolchain <<EOF
${LEAN_TOOLCHAIN}
EOF

RUN lake update && lake exe cache get && lake env printenv LEAN_PATH > /workspace/.yubo_lean_path

USER root
RUN cat > /usr/local/bin/yubo-lean <<'EOF'
#!/usr/bin/env sh
set -eu
cd /workspace
if [ -f /workspace/.yubo_lean_path ]; then
  yubo_lean_path="$(cat /workspace/.yubo_lean_path)"
  if [ -n "${LEAN_PATH:-}" ]; then
    export LEAN_PATH="${yubo_lean_path}:${LEAN_PATH}"
  else
    export LEAN_PATH="${yubo_lean_path}"
  fi
fi
exec lean "$@"
EOF
RUN chmod +x /usr/local/bin/yubo-lean

USER lean
RUN printf '%s\n' 'import Mathlib' '' 'example : 1 + 1 = 2 := by norm_num' > /tmp/yubo_preflight.lean \
  && yubo-lean /tmp/yubo_preflight.lean
