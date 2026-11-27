PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: install lint format format-check typecheck test check precommit install-hooks run

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

precommit:
	@if [ "$$(git config --local --get core.hooksPath)" = ".githooks" ]; then git config --local --unset-all core.hooksPath; fi
	@python_path="$$( $(PYTHON) -c 'import sys; print(sys.executable)' )"; \
	cd "$$(git rev-parse --show-toplevel)" && \
	"$$python_path" -m pre_commit install --config lab_1/ws-http-server/.pre-commit-config.yaml --allow-missing-config --overwrite

install-hooks: precommit

run:
	$(PYTHON) main.py
