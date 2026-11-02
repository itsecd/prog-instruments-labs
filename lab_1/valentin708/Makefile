PYTHON ?= python3
ANNOTATION ?= annotation.csv
OUTPUT_DIR ?= .

.PHONY: run lint format format-check typecheck fullcheck precommit

run:
	MPLBACKEND=Agg $(PYTHON) main.py --annotation "$(ANNOTATION)" --output-dir "$(OUTPUT_DIR)"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright --project pyrightconfig.json

fullcheck: lint format-check typecheck

precommit:
	cd ../.. && $(PYTHON) -m pre_commit install --config lab_1/valentin708/.pre-commit-config.yaml --hook-type pre-commit --hook-type pre-push
