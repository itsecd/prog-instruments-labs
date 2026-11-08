.PHONY: lint format format-check typecheck fullcheck precommit
lint:
	ruff check


format:
	ruff format


format-check:
	ruff format --check


typecheck:
	pyright


fullcheck: format-check lint typecheck
precommit:
	pre-commit install
	pre-commit install --hook-type pre-push