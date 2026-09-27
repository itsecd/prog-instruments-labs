.PHONY: lint format typecheck fullcheck test fulltest precommit

lint:
	ruff check .

format:
	ruff format .

typecheck:
	pyright

fullcheck: lint typecheck

test:
	pytest

fulltest: fullcheck test

precommit:
	pre-commit install