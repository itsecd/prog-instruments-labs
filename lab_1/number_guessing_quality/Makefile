PYTHON ?= python

.PHONY: run lint format format-check typecheck test fullcheck fulltest precommit

run:
	$(PYTHON) game.py

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright .

test:
	$(PYTHON) -m unittest discover -s tests -v

fullcheck: lint format-check typecheck

fulltest: fullcheck test

precommit:
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type pre-push
