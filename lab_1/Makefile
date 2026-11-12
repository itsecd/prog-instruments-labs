ifeq ($(OS),Windows_NT)
	PYTHON := py
else
	PYTHON := python3
endif

.PHONY: run lint format format-check typecheck fullcheck test fulltest precommit
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

precommit:
	cd .. && $(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type pre-push -c lab_1/.pre-commit-config.yaml