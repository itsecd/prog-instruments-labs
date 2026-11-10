.PHONY: run lint format typecheck fullcheck precommit

run:
	python main.py data.txt

lint:
	ruff check .

format:
	ruff format .

typecheck:
	pyright .

fullcheck: lint typecheck

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push