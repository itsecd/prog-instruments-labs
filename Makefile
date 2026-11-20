.PHONY: run lint format check-format typecheck test fullcheck precommit


run:
	python3 lab_1/task_manager.py


format:
	ruff format lab_1


check-format:
	ruff format --check lab_1


lint:
	ruff check lab_1


lint-fix:
	ruff check --fix lab_1


typecheck:
	pyright


test:
	pytest lab_1


fullcheck: format lint typecheck


precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push