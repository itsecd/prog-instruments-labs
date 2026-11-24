.PHONY: install run lint format format-check typecheck fullcheck precommit

install:
	pip install -r requirements.txt

run:
	python code.py data.txt

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	pyright

fullcheck: lint format-check typecheck

precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push