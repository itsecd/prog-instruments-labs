PYTHON ?= python

.PHONY: lint format format-check typecheck fullcheck precommit

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright --project pyproject.toml

fullcheck: lint format-check typecheck

precommit:
	$(PYTHON) -m pre_commit install --config lab_1/.pre-commit-config.yaml --allow-missing-config --install-hooks
