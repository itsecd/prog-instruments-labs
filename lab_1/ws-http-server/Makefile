PYTHON ?= python3

.PHONY: install lint format format-check typecheck test check install-hooks run

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright

test:
	$(PYTHON) -m pytest -q

check: lint format-check typecheck test

install-hooks:
	git config core.hooksPath .githooks

run:
	$(PYTHON) main.py
