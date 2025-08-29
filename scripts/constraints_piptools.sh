#!/usr/bin/env bash
set -euo pipefail

# Generate fully pinned constraints.txt from dev.in using pip-tools.
# Requires: pip install -e ".[dev]" to have pip-compile available.
# Usage: ./scripts/constraints_piptools.sh [--upgrade]

ARGS=""
if [[ "${1:-}" == "--upgrade" ]]; then
  ARGS="--upgrade"
fi

if ! command -v pip-compile >/dev/null 2>&1; then
  echo "pip-compile not found. Run: pip install -e \".[dev]\" or pip install pip-tools"
  exit 1
fi

pip-compile dev.in --output-file constraints.txt ${ARGS}
echo "Generated constraints.txt"
