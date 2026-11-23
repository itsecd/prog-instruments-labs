.PHONY: lint format format-check typecheck test fullcheck fulltest precommit clean help

PYTHON ?= python
PYTEST ?= pytest

help:
	@echo "Доступные команды:"
	@echo "  make lint         - статический анализ (ruff check)"
	@echo "  make format       - автоформатирование кода (ruff format)"
	@echo "  make format-check - проверка форматирования (ruff format --check)"
	@echo "  make typecheck    - проверка типов (pyright)"
	@echo "  make test         - запуск тестов (pytest)"
	@echo "  make fullcheck    - lint + format-check + typecheck"
	@echo "  make fulltest     - fullcheck + test"
	@echo "  make precommit    - установка pre-commit и pre-push хуков"
	@echo "  make clean        - очистка временных файлов и кэшей"

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	pyright

test:
	$(PYTEST); code=$$?; [ $$code -eq 0 ] || [ $$code -eq 5 ]

fullcheck: lint format-check typecheck

fulltest: fullcheck test

precommit:
	pre-commit install --install-hooks
	pre-commit install --hook-type pre-push

clean:
	rm -rf .ruff_cache .pytest_cache __pycache__ A-*.txt B-*.txt C-*.txt results.csv plot.png 