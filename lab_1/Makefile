ifeq ($(OS),Windows_NT)
	PYTHON := py
else
	PYTHON := python3
endif

.PHONY: run lint format format-check typecheck fullcheck test fulltest

run:
	$(PYTHON) -m src.main

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright

fullcheck: lint format-check typecheck

test:
	$(PYTHON) -m unittest discover -s tests -p "Test*.py"

fulltest: fullcheck test