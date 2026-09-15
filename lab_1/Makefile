.PHONY: install lint format typecheck test fullcheck run

install:
	pip install -r requirements.txt
	pip install ruff pyright pre-commit pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	pyright

test:
	pytest

fullcheck: lint typecheck test

run:
	python main.py

precommit:
	pre-commit install