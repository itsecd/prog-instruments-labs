PYTHON ?= python
MAIN := lab_1.py
SRC := .
TESTS := .

.PHONY: help install run lint lint-fix format format-check typecheck fullcheck test fulltest precommit

help:
	@echo "make install      - install dev dependencies"
	@echo "make run          - run the program"
	@echo "make lint         - run ruff linter"
	@echo "make lint-fix     - run ruff linter with auto-fix"
	@echo "make format       - format code with ruff"
	@echo "make format-check - check code formatting"
	@echo "make typecheck    - run pyright"
	@echo "make fullcheck    - lint, format check and typecheck"
	@echo "make test         - run pytest"
	@echo "make fulltest     - fullcheck and tests"
	@echo "make precommit    - create pre-commit hooks"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) $(MAIN)

lint:
	$(PYTHON) -m ruff check $(SRC)

lint-fix:
	$(PYTHON) -m ruff check $(SRC) --fix

format:
	$(PYTHON) -m ruff format $(SRC)

format-check:
	$(PYTHON) -m ruff format $(SRC) --check

typecheck:
	$(PYTHON) -m pyright $(SRC)

fullcheck: lint format-check typecheck

test:
	$(PYTHON) -m pytest $(TESTS) -v

fulltest: fullcheck test

precommit:
	$(PYTHON) -m pre_commit install --config .pre-commit-config.yaml --hook-type pre-commit --hook-type pre-push