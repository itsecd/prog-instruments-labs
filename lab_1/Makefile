.PHONY: lint format format-check typecheck fullcheck precommit test

lint:
	ruff check task_1 task_2

format:
	ruff format task_1 task_2

format-check:
	ruff format --check task_1 task_2

typecheck:
	pyright task_1 task_2

fullcheck: lint format typecheck

precommit:
	pre-commit install

test:
	@echo "No tests configured yet"
