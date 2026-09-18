PYTHON = python
PIP = $(PYTHON) -m pip
VENV = .venv
PYTHON_VENV = $(VENV)/Scripts/python.exe
PIP_VENV = $(PYTHON_VENV) -m pip
SRC_DIR = src
TEST_DIR = tests

# Объявляем цели, которые не являются именами файлов
.PHONY: help install activate run test lint format typecheck fullcheck fulltest clean

# Цель по умолчанию (выполняется при простом вводе 'make')
help:
	@echo "Доступные команды для проекта:"
	@echo ""
	@echo "make install     - Создать .venv и установить все зависимости"
	@echo "make activate    - Показать команду активации виртуального окружения"
	@echo "make run         - Запустить основной скрипт (ежедневный прогноз)"
	@echo "make test        - Запустить тесты (pytest)"
	@echo "make lint        - Проверить стиль кода и исправить (Ruff)"
	@echo "make format      - Отформатировать код (Ruff)"
	@echo "make typecheck   - Проверить типы (Pyright)"
	@echo "make fullcheck   - Запустить все статические проверки (lint + typecheck)"
	@echo "make fulltest    - Запустить статические проверки + все тесты"
	@echo "make full        - Зависимости + запуск основного скрипта + все проверки"
	@echo ""

# Создание виртуального окружения и установка зависимостей
install:
	@echo "Создание виртуального окружения..."
	$(PYTHON) -m venv $(VENV)
	@echo "Обновление pip..."
	$(PYTHON_VENV) -m pip install --upgrade pip
	@echo "Установка зависимостей из requirements.txt..."
	$(PYTHON_VENV) -m pip install -r requirements.txt
	@echo "Установка инструментов разработки (Ruff, Pyright, Pytest)..."
	$(PYTHON_VENV) -m pip install ruff pyright pytest approvaltests
	@echo ""
	@echo "Готово! Активируйте окружение командой:"
	@echo "$(VENV)\\Scripts\\activate"

# Показать команду активации окружения
activate:
	@echo "Для активации окружения выполните:"
	@echo "$(VENV)\\Scripts\\activate"

# Запуск основного скрипта (ежедневный прогноз)
run:
	@echo "Запуск ежедневного прогноза..."
	@echo "Внимание: для работы lookup_weather() нужен сервер на http://127.0.0.1:3005"
	@echo "Если сервер не запущен, программа завершится с RuntimeError."
	$(PYTHON) -c "import sys; sys.path.append('$(SRC_DIR)'); import daily_forecasts; daily_forecasts.print_sales_forecasts()"

# Запуск всех тестов
test:
	@echo "Запуск тестов (pytest)..."
	$(PYTHON_VENV) -m pytest $(TEST_DIR) -v --tb=short

# Проверка стиля кода (без исправления)
lint:
	@echo "Проверка стиля кода (Ruff)..."
	$(PYTHON_VENV) -m ruff check $(SRC_DIR) $(TEST_DIR)
	$(PYTHON_VENV) -m ruff check --fix $(SRC_DIR) $(TEST_DIR)

# Автоматическое форматирование кода
format:
	@echo "Форматирование кода и автоисправление..."
	$(PYTHON_VENV) -m ruff format $(SRC_DIR) $(TEST_DIR)
	$(PYTHON_VENV) -m ruff check --fix $(SRC_DIR) $(TEST_DIR)

# Проверка аннотаций типов
typecheck:
	@echo "Проверка типов (Pyright)..."
	$(PYTHON_VENV) -m pyright $(SRC_DIR)

# Все статические проверки (линтер + форматтер + типы)
fullcheck: format lint typecheck
	@echo "Все статические проверки пройдены!"

# Статические проверки + тесты
fulltest: fullcheck test
	@echo "Все проверки и тесты пройдены успешно!"

full: install activate run fulltest
	@echo "Все проверено, все работает"
# Установка pre-commit hooks в .git/hooks/
precommit:
	@echo "🔧 Установка pre-commit hooks..."
	$(PYTHON_VENV) -m pip install pre-commit
	$(PYTHON_VENV) -m pre_commit install --hook-type pre-commit
	$(PYTHON_VENV) -m pre_commit install --hook-type pre-push
	@echo "✅ Hooks установлены!"
	@echo "   - pre-commit: линтинг, форматирование, проверка типов"
	@echo "   - pre-push: запуск всех тестов"

# Удаление pre-commit hooks
precommit-uninstall:
	@echo "🗑️  Удаление pre-commit hooks..."
	$(PYTHON_VENV) -m pre_commit uninstall --hook-type pre-commit
	$(PYTHON_VENV) -m pre_commit uninstall --hook-type pre-push
	@echo "✅ Hooks удалены!"