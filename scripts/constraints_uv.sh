#!/usr/bin/env bash
set -euo pipefail

# Generate fully pinned constraints.txt from dev.in using uv.
# Requires: pip install -e ".[dev]" to have uv available (or pip install uv).
# Usage: ./scripts/constraints_uv.sh

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Run: pip install -e \".[dev]\" or pip install uv"
  exit 1
fi

uv pip compile dev.in -o constraints.txt
echo "Generated constraints.txt"
