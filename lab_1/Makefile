.PHONY: help install lint format typecheck test fullcheck fulltest precommit

PYTHON ?= python

help:
	@echo "Доступные команды:"
	@echo "  make install    - Установить зависимости"
	@echo "  make lint       - Запустить линтер (Ruff)"
	@echo "  make format     - Отформатировать код"
	@echo "  make typecheck  - Проверить типы (Pyright)"
	@echo "  make test       - Запустить тесты (Pytest)"
	@echo "  make fullcheck  - Проверить форматирование, линтер и типы"
	@echo "  make fulltest   - Все проверки + запуск тестов"
	@echo "  make precommit  - Установить git-хуки"

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m pyright .

test:
	$(PYTHON) -m pytest .

fullcheck:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m pyright .

fulltest: fullcheck test

precommit:
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type pre-push