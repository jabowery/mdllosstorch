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
