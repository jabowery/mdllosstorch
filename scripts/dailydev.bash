#!/bin/bash

# from repo root
. .venv/bin/activate            # activate your venv
pip install -e . -r requirements-dev.txt
pre-commit install              # one-time, then it runs on every commit

pytest -q                       # run tests
make lint && make format        # tidy code
make typecheck                  # mypy (optional)
