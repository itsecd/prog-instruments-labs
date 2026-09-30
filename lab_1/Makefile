.DEFAULT_GOAL := help
PYTHON ?= python
ARGS ?= --help

.PHONY: help install run lint lint-fix format format-check typecheck fullcheck test fulltest build precommit hooks-check

help:
	@echo "install       Install project and development tools"
	@echo "run           Run CLI (pass ARGS='--input ... --output ... --width ... --height ...')"
	@echo "lint          Check code with Ruff"
	@echo "lint-fix      Apply safe Ruff fixes"
	@echo "format        Format Python files"
	@echo "format-check  Check formatting without changing files"
	@echo "typecheck     Run Pyright in strict mode"
	@echo "fullcheck     Run all static checks"
	@echo "test          Run tests"
	@echo "fulltest      Run static checks and tests"
	@echo "build         Build wheel and source distribution"
	@echo "precommit     Install pre-commit and pre-push hooks"
	@echo "hooks-check   Run both hook stages on all files"

install:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) -m crop_cli $(ARGS)

lint:
	$(PYTHON) -m ruff check .

lint-fix:
	$(PYTHON) -m ruff check --fix .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright

fullcheck: lint format-check typecheck

test:
	$(PYTHON) -m pytest

fulltest: fullcheck test

build:
	$(PYTHON) -m build

precommit:
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type pre-push

hooks-check:
	$(PYTHON) -m pre_commit run --all-files --hook-stage pre-commit
	$(PYTHON) -m pre_commit run --all-files --hook-stage pre-push
