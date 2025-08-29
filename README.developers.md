
# mdllosstorch – Instructions for Developers

You are a **developer** (contributor). You want to *clone the repo, run tests, and hack*.

## Setup
```bash
git clone https://github.com/yourusername/mdllosstorch.git
cd mdllosstorch

# install into a venv with dev extras
./scripts/dev_install.sh
```

## Updating constraints
Constraints ensure reproducibility across contributors and CI.

```bash
# Update using uv (preferred) or pip-tools
./scripts/dev_update_constraints.sh --upgrade

# Commit updated constraints.txt
git add constraints.txt
git commit -m "chore: update constraints"
```

## Running tests and tools
```bash
pytest -q          # run tests
ruff check src     # lint
black src tests    # format
mypy src           # type-check
pre-commit run --all-files
```

## Verifying environment
```bash
./scripts/dev_verify.sh
```

## Docs
```bash
./scripts/mkdocs.sh serve
```

## Notes
- Runtime dependencies: see `[project.dependencies]` in `pyproject.toml`
- Dev/test/docs dependencies: see `[project.optional-dependencies]` in `pyproject.toml`
- Constraints (`constraints.txt`) are generated from `dev.in` using uv or pip-tools
