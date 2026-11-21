.PHONY: run test lint format format-check typecheck fullcheck fulltest precommit clean

run:
	python lab1.py data.txt -o cities_stats.txt

test:
	python -m unittest discover -s . -p "test_*.py" -v

lint:
	python -m ruff check .

format:
	python -m ruff format .
	python -m ruff check --fix .

format-check:
	python -m ruff format --check .

typecheck:
	python -m pyright

fullcheck: lint format-check typecheck

fulltest: fullcheck test

precommit:
	python -m pre_commit install -c .pre-commit-config.yaml --hook-type pre-commit --hook-type pre-push

clean:
	rm -f cities_stats.txt