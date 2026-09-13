.PHONY: install lint format format-check typecheck fullcheck test fulltest run precommit

install:
	pip install ruff pyright pytest pre-commit

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	pyright

fullcheck: lint format-check typecheck

test:
	pytest

fulltest: fullcheck test

run:
	python -c "from yahtzee import Yahtzee; print(Yahtzee.chance(1,2,3,4,5))"

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push