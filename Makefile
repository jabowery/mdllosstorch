.PHONY: dev test lint format typecheck build clean

VENV?=.venv
PY?=python3

dev:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -e . -r requirements-dev.txt
	. $(VENV)/bin/activate && pre-commit install

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests
	ruff format src tests

typecheck:
	mypy src

build:
	python -m build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache src/*.egg-info dist build $(VENV)


# --- Census integration helpers ---------------------------------------------

.PHONY: census-install census-test census-run

# Install dev + census extras (uses constraints.txt if present)
census-install:
	@if [ -f constraints.txt ]; then \
		pip install -e ".[dev,census]" -c constraints.txt ; \
	else \
		pip install -e ".[dev,census]"; \
	fi

# Run only the census integration test (will SKIP if dataset env var not set)
census-test:
	pytest -q tests/test_census_integration.py

# CLI runner for ad-hoc exploration (accepts --rows/--cols/--method)
# Example:
#   make census-run ARGS="--rows 10000 --cols 1024 --method yeo-johnson"
census-run:
	python scripts/run_census_mdl.py $(ARGS)

.PHONY: census-test-verbose
census-test-verbose:
	@LOTC_VERBOSE=1 pytest -q tests/test_census_integration.py

