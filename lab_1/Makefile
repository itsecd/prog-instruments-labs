.PHONY: install run lint format format-check typecheck fullcheck test fulltest precommit

install:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	python lab1.py

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

format-check:
	ruff format --check .

typecheck:
	pyright

fullcheck: lint format-check typecheck

test:
	pytest

fulltest: fullcheck test

precommit:
	cd .. && pre-commit install -c lab_1/.pre-commit-config.yaml