.PHONY: lint format typecheck fullcheck precommit


lint:
	ruff check


format:
	ruff format


typecheck:
	pyright


fullcheck: format lint typecheck


precommit:
	pre-commit install