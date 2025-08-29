#!/usr/bin/env bash
set -euo pipefail

# Helper to regenerate constraints using whichever tool is available (uv preferred).
# Usage: ./scripts/dev_update_constraints.sh [--upgrade]

if command -v uv >/dev/null 2>&1; then
  ./scripts/constraints_uv.sh "${1:-}"
elif command -v pip-compile >/dev/null 2>&1; then
  ./scripts/constraints_piptools.sh "${1:-}"
else
  echo "Neither uv nor pip-compile is available. Install dev extras first:"
  echo "  pip install -e \".[dev]\""
  exit 1
fi
