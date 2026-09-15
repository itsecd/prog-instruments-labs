.PHONY: help install run lint format format-check typecheck fullcheck test fulltest precommit clean

help:
	@echo Available commands:
	@echo   make install       - install dependencies
	@echo   make run           - run the script on test data
	@echo   make lint          - run linter
	@echo   make format        - format code
	@echo   make format-check  - check formatting
	@echo   make typecheck     - run type checker
	@echo   make fullcheck     - all static checks
	@echo   make test          - run tests
	@echo   make fulltest      - checks + tests
	@echo   make precommit     - install git hooks
	@echo   make clean         - clean caches

install:
	pip install -r requirements.txt

run:
	python lab1_var13.py data.txt result.txt

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format . --check

typecheck:
	pyright

fullcheck: lint typecheck format-check

test:
	pytest -v

fulltest: fullcheck test

precommit:
	pre-commit install --config .pre-commit-config.yaml
	pre-commit install --hook-type pre-push --config .pre-commit-config.yaml

clean:
	powershell -Command "Get-ChildItem -Recurse -Directory -Include __pycache__,.pytest_cache,.ruff_cache | Remove-Item -Recurse -Force"
