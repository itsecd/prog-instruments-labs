.PHONY: help install lint format format-check typecheck test fullcheck fulltest precommit run

help:
	@echo "Доступные команды:"
	@echo "  make install       - установка зависимостей"
	@echo "  make lint          - статический анализ (ruff check)"
	@echo "  make format        - автоформатирование (ruff format)"
	@echo "  make format-check  - проверка форматирования"
	@echo "  make typecheck     - проверка типов (pyright)"
	@echo "  make test          - запуск тестов"
	@echo "  make fullcheck     - все статические проверки"
	@echo "  make fulltest      - статические проверки + тесты"
	@echo "  make precommit     - установка git hooks"
	@echo "  make run           - запуск калькулятора"

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

test:
	pytest -v

fullcheck: lint format-check typecheck

fulltest: fullcheck test

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push

run:
	python -m calc.cli $(ARGS)