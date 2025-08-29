#!/usr/bin/env bash
set -euo pipefail

# This script exercises pyproject.toml for all roles:
# - Users: install runtime only, import smoke test
# - Developers: install dev extras, run tests, (optionally) generate constraints
# - Maintainers: build & twine check, verify metadata, docs build from extras

PROJ_ROOT="${PROJ_ROOT:-$(pwd)}"
VENV=".venv-soak"

echo "===> Using project root: $PROJ_ROOT"

# fresh venv
python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip wheel

echo "===> USER role: runtime-only install and import"
pip install .
python - <<'PY'
import importlib, sys
m = importlib.import_module("mdllosstorch")
print("import ok; version:", getattr(m, "__version__", "n/a"))
PY

echo "===> Clean runtime install"
pip uninstall -y mdllosstorch || true

echo "===> DEVELOPER role: editable install with dev extras"
pip install -e ".[dev]"

echo "===> Run tests"
pytest -q

echo "===> Try generating constraints (uv preferred, else pip-compile)"
if command -v uv >/dev/null 2>&1; then
  echo "(uv) compiling dev.in -> constraints.txt"
  uv pip compile dev.in -o constraints.txt
elif command -v pip-compile >/dev/null 2>&1; then
  echo "(pip-tools) compiling dev.in -> constraints.txt"
  pip-compile dev.in --output-file constraints.txt
else
  echo "No uv or pip-compile available; skipping constraints generation."
fi

echo "===> MAINTAINER role: build sdist & wheel"
python -m build

echo "===> Twine metadata check"
python -m pip install -U twine
python -m twine check dist/*

echo "===> Metadata sanity checks"
python scripts/check_metadata.py

echo "===> DOCS extras smoke test (build site)"
pip install ".[docs]" || true
if command -v mkdocs >/dev/null 2>&1; then
  mkdocs build || echo "mkdocs build failed (ok if docs not configured)"
else
  echo "mkdocs not installed; skipping docs build"
fi

echo "===> Done. Artifacts:"
ls -al dist || true
