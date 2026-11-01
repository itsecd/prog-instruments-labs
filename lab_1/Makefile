.PHONY: help install run lint format format-check typecheck test fullcheck precommit

help:
	@echo "install       - установить зависимости"
	@echo "run           - запустить обе задачи"
	@echo "lint          - ruff check"
	@echo "format        - ruff format"
	@echo "format-check  - ruff format --check"
	@echo "typecheck     - pyright"
	@echo "test          - pytest"
	@echo "fullcheck     - lint + format-check + typecheck"
	@echo "precommit     - установить git hooks"

install:
	pip install -r requirements.txt

run:
	python task1/task1.py
	python task2/task2.py

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	pyright .

test:
	pytest

fullcheck: lint format-check typecheck

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push
