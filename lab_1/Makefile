
.PHONY: check lint format format-check typecheck run

check: lint format-check typecheck
lint: ; .venv/bin/ruff check --config ruff.toml expense_tracker.py
format: ; .venv/bin/ruff format --config ruff.toml expense_tracker.py
format-check: ; .venv/bin/ruff format --check --config ruff.toml expense_tracker.py
typecheck: ; .venv/bin/pyright --project pyrightconfig.json
run: ; .venv/bin/python expense_tracker.py --help


.PHONY: precommit
precommit: ; cd .. && lab_1/.venv/bin/pre-commit install -t pre-commit -t pre-push --config lab_1/.pre-commit-config.yaml
