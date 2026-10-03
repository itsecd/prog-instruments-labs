PYTHON := py
RUFF := $(PYTHON) -m ruff
PYRIGHT := $(PYTHON) -m pyright
PRECOMMIT := $(PYTHON) -m pre_commit

.PHONY: help lint format format-check typecheck fullcheck test fulltest run run-gui precommit clean

help:
	@echo "Доступные команды:"
	@echo "  make lint          - запуск ruff check"
	@echo "  make format        - автоформатирование ruff format"
	@echo "  make format-check  - проверка форматирования без изменений"
	@echo "  make typecheck     - проверка типов pyright"
	@echo "  make fullcheck     - все статические проверки (lint + format-check + typecheck)"
	@echo "  make test          - запуск unit-тестов"
	@echo "  make fulltest      - все проверки + тесты"
	@echo "  make run           - запуск CLI"
	@echo "  make run-gui       - запуск GUI"
	@echo "  make precommit     - установка git-хуков"
	@echo "  make clean         - удалить кэши"

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format . --check

typecheck:
	$(PYRIGHT)

fullcheck: lint format-check typecheck

test:
	$(PYTHON) -m unittest test_hash -v

fulltest: fullcheck test

run:
	$(PYTHON) cli.py --help

run-gui:
	$(PYTHON) gui.py

precommit:
	$(PRECOMMIT) install
	$(PRECOMMIT) install --hook-type pre-push

clean:
	@echo "Очистка кэшей..."
	-rmdir /s /q __pycache__ 2>nul || true
	-rmdir /s /q .ruff_cache 2>nul || true
	-rmdir /s /q .pytest_cache 2>nul || true