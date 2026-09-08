#!/usr/bin/env bash
set -euo pipefail

runtime="${THM_DOCKER_RUNTIME:-docker}"
lang="${1:-lean4}"
case "$lang" in
  lean4)
    image="${THM_LEAN4_DOCKER_IMAGE:-yubo-lean4-mathlib:latest}"
    echo "Lean 4 image builds are not defined in this repository." >&2
    echo "Provide an offline-ready image with THM_LEAN4_DOCKER_IMAGE; current image: $image" >&2
    exit 1
    ;;
  coq)
    image="${THM_COQ_DOCKER_IMAGE:-coqorg/coq:8.19}"
    echo "No custom build required for coq; using image: $image"
    ;;
  isabelle)
    image="${THM_ISABELLE_DOCKER_IMAGE:-makarius/isabelle:latest}"
    echo "No custom build required for isabelle; using image: $image"
    ;;
  *)
    echo "Unsupported theorem language: $lang" >&2
    exit 2
    ;;
esac
