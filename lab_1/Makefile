.PHONY: install lint format format-check typecheck fullcheck test fulltest run precommit

install:
	python -m pip install -r requirements.txt
	python -m pip install ruff pyright pre-commit pytest

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
	python main.py -c red green

precommit:
	pre-commit install --config ../.pre-commit-config.yaml --hook-type pre-commit --hook-type pre-push