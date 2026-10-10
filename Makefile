.PHONY: lint format typecheck fullcheck test fulltest precommit run

lint:
	ruff check .

format:
	ruff format .
	ruff check --select I --fix .

typecheck:
	pyright .

fullcheck: format lint typecheck

test:
	pytest .

fulltest: fullcheck test

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push

run:
	python main.py --config config.json