PYTHON ?= python
PRECOMMIT_CONFIG := .pre-commit-config.yaml

INPUT ?= data/input.wav
OUTPUT ?= output/trimmed.wav
START ?= 0
END ?= 5

.DEFAULT_GOAL := help
.PHONY: help install run lint lint-fix format format-check typecheck test \
        fullcheck fulltest precommit precommit-run clean

help:
	@$(PYTHON) -c "import re; [print(f'{m[1]:<15} {m[2]}') for m in re.finditer(r'^([a-z-]+):.*?## (.*)', open('Makefile', encoding='utf-8').read(), re.M)]"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py --input $(INPUT) --output $(OUTPUT) --start $(START) --end $(END)

lint:
	$(PYTHON) -m ruff check .

lint-fix:
	$(PYTHON) -m ruff check --fix .

format:
	$(PYTHON) -m ruff check --select I --fix .
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m pyright

test:
	$(PYTHON) -m pytest

fullcheck: lint format-check typecheck

fulltest: fullcheck test

precommit:
	$(PYTHON) -m pre_commit install --config $(PRECOMMIT_CONFIG) --hook-type pre-commit --hook-type pre-push

precommit-run:
	$(PYTHON) -m pre_commit run --config $(PRECOMMIT_CONFIG) --all-files
	$(PYTHON) -m pre_commit run --config $(PRECOMMIT_CONFIG) --all-files --hook-stage pre-push

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('*')) if p.name in ('__pycache__', '.pytest_cache', '.ruff_cache') and '.venv' not in p.parts]"
