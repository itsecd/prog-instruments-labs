.DEFAULT_GOAL := fullcheck
.RECIPEPREFIX := >

ifeq ($(OS),Windows_NT)
PYTHON ?= ../.venv/Scripts/python.exe
else
PYTHON ?= python3
endif

PYTHON_EXE := $(shell "$(PYTHON)" -c "import sys; print(sys.executable)")

.PHONY: run gui lint format format-check typecheck test fullcheck fulltest

run:
>cd project && "$(PYTHON_EXE)" main.py cli --bits 8 --experiments 1

gui:
>cd project && "$(PYTHON_EXE)" main.py gui

lint:
>"$(PYTHON_EXE)" -m ruff check project

format:
>"$(PYTHON_EXE)" -m ruff format project

format-check:
>"$(PYTHON_EXE)" -m ruff format --check project

typecheck:
>"$(PYTHON_EXE)" -m pyright --project pyproject.toml --pythonpath "$(PYTHON_EXE)" --warnings

test:
>cd project && "$(PYTHON_EXE)" -m unittest discover -s tests -v

fullcheck: lint format-check typecheck

fulltest: fullcheck test

.PHONY: precommit

precommit:
>cd .. && "$(PYTHON_EXE)" -m pre_commit install --hook-type pre-commit --hook-type pre-push