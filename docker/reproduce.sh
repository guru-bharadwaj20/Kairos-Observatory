#!/usr/bin/env bash
# Entry point of the replication container (K-12.07).
# Runs the validation pipeline and copies the report and figures to /out.
set -euo pipefail

out="${KAIROS_OUT:-/out}"
mkdir -p "$out"

args=(validation --strict --root /work --report "$out/validation-run.json")
if [ "${KAIROS_CPU_ONLY:-0}" = "1" ]; then
  args+=(--cpu-only)
else
  uv run --no-sync kairos doctor || echo "warning: kairos doctor reported a problem" >&2
fi

uv run --no-sync python -m kairos.commands.run "${args[@]}" "$@"

cp -v docs/validation.md "$out/"
if [ -d docs/figures ]; then cp -rv docs/figures "$out/"; fi
sha256sum "$out/validation.md" | tee "$out/validation.md.sha256"
