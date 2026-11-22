PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
SOURCE := main.py

.PHONY: setup run lint format format-check typecheck fullcheck precommit

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -r requirements.txt

run:
	$(BIN)/python $(SOURCE)

lint:
	$(BIN)/ruff check $(SOURCE)

format:
	$(BIN)/ruff format $(SOURCE)

format-check:
	$(BIN)/ruff format --check $(SOURCE)

typecheck:
	$(BIN)/pyright --pythonpath $(BIN)/python

fullcheck: lint format-check typecheck

precommit:
	$(BIN)/pre-commit install
