.PHONY: lint format typecheck fullcheck test fulltest precommit run

lint:
	ruff check .

format:
	ruff format .

typecheck:
	pyright .

fullcheck: lint typecheck
	ruff format . --check

test:
	pytest . -v

fulltest: fullcheck test

precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push

run:
	python texttest_fixture.py
