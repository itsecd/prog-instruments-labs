.PHONY: help install lint format typecheck test check fullcheck precommit run

PYTHON = python
RUFF = ruff
PYRIGHT = pyright
PYTEST = pytest

help:
	@echo "Available commands:"
	@echo "  make run        - Run main application"
	@echo "  make lint       - Check code with Ruff linter"
	@echo "  make format     - Automatically format code with Ruff"
	@echo "  make typecheck  - Perform static type checking with Pyright"
	@echo "  make test       - Run unit tests with Pytest"
	@echo "  make fullcheck  - Run all static checks and tests"
	@echo "  make precommit  - Install Git pre-commit and pre-push hooks"

run:
	$(PYTHON) lab_1.py

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

typecheck:
	$(PYRIGHT)

test:
	$(PYTEST)

fullcheck: lint format typecheck test

precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push