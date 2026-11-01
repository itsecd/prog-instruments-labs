VENV   := .venv
BIN    := $(VENV)/bin
PYTHON ?= python3

.PHONY: help install lint lint-fix format format-check typecheck fullcheck \
        test fulltest precommit clean

help:  ## Show available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  make %-13s %s\n", $$1, $$2}'

$(BIN)/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	touch $(BIN)/activate

install: $(BIN)/activate  ## Create .venv and install dependencies

lint: install  ## Run linter (ruff check)
	$(BIN)/ruff check .

lint-fix: install  ## Run linter and auto-fix what is possible
	$(BIN)/ruff check --fix .

format: install  ## Format code (ruff format)
	$(BIN)/ruff format .

format-check: install  ## Check formatting without changing files
	$(BIN)/ruff format --check .

typecheck: install  ## Run type checker (pyright)
	$(BIN)/pyright

fullcheck: lint format-check typecheck  ## Run all static checks

test: install  ## Run tests (pytest)
	$(BIN)/pytest

fulltest: fullcheck test  ## Run all static checks and tests

precommit: install  ## Install git hooks (pre-commit and pre-push)
	$(BIN)/pre-commit install --config .pre-commit-config.yaml \
		--hook-type pre-commit --hook-type pre-push

clean:  ## Remove virtualenv and caches
	rm -rf $(VENV) .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
