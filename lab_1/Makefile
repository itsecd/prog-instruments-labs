.PHONY: run tests format check types fix format_fix fullcheck fullfix precommit install

install:
	python -m pip install -r requirements.txt

run:
	python main.py input.txt

tests:
	python -m pytest tests/

format:
	ruff format . --check

check:
	ruff check .

types:
	pyright --project pyproject.toml

fix:
	ruff check . --fix

format_fix:
	ruff format .

fullcheck: tests check format types

fullfix: fix format_fix

precommit:
	cd .. && pre-commit install --config lab_1/.pre-commit-config.yaml --hook-type pre-commit --hook-type pre-push