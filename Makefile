.PHONY: lint format typecheck fullcheck test precommit

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	pyright

fullcheck: format typecheck

precommit:
	pre-commit install
