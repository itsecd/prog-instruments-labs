.PHONY: help install lint format typecheck test fulltest precommit

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make lint       - Run linter (ruff check)"
	@echo "  make format     - Format code (ruff format)"
	@echo "  make typecheck  - Type check (pyright)"
	@echo "  make test       - Run tests (pytest)"
	@echo "  make fulltest   - Run all checks and tests"
	@echo "  make precommit  - Install pre-commit hooks"

install:
	pip install -r req.txt

lint:
	ruff check lab_1

format:
	ruff format lab_1

typecheck:
	pyright lab_1

test:
	pytest lab_1/tests

fulltest: lint format typecheck test

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push