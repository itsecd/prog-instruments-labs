.PHONY: help install lint format format-check typecheck fullcheck run test fulltest precommit clean

PYTHON := python
RUFF := ruff
PYRIGHT := pyright

help:
	@echo "Доступные команды:"
	@echo "  make install       - установка зависимостей"
	@echo "  make lint          - запуск линтера Ruff"
	@echo "  make format        - форматирование кода"
	@echo "  make format-check  - проверка форматирования"
	@echo "  make typecheck     - проверка типов Pyright"
	@echo "  make fullcheck     - все статические проверки"
	@echo "  make run           - запуск программы"
	@echo "  make test          - запуск тестов"
	@echo "  make fulltest      - все проверки и тесты"
	@echo "  make precommit     - установка git-хуков"
	@echo "  make clean         - очистка кэша"

install:
	pip install ruff pyright pre-commit pytest

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

typecheck:
	$(PYRIGHT)

fullcheck: lint format-check typecheck
	@echo "Все проверки пройдены!"

run:
	$(PYTHON) uno.py data.txt result.txt

test:
	$(PYTHON) -m pytest tests/ -v

fulltest: fullcheck test

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +