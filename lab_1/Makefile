.PHONY: run lint format typecheck fullcheck precommit

run:
	python task1.py task1.txt key.txt out.txt

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	pyright

fullcheck: lint typecheck
	ruff format --check .

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push
