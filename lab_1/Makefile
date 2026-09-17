.PHONY: help venv install run lint format format-check typecheck fullcheck test fulltest build deploy precommit clean

PYTHON := venv/Scripts/python.exe

help:
	@echo Доступные команды:
	@echo   make venv          - Создать виртуальное окружение
	@echo   make install       - Установить зависимости
	@echo   make run           - Запуск программы
	@echo   make lint          - Проверка кода линтером (Ruff)
	@echo   make format        - Автоформатирование кода (Ruff)
	@echo   make format-check  - Проверка форматирования (Ruff)
	@echo   make typecheck     - Проверка типов (Pyright)
	@echo   make fullcheck     - Все статические проверки
	@echo   make test          - Запуск тестов
	@echo   make fulltest      - Статические проверки + тесты
	@echo   make build         - Сборка пакета
	@echo   make deploy        - Заглушка деплоя
	@echo   make precommit     - Установка Git hooks
	@echo   make clean         - Очистка временных файлов

venv:
	$(PYTHON) -m venv venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install ruff pyright pre-commit pytest build

run:
	$(PYTHON) script.py text.txt alfa.txt key.txt decrypted.txt

lint:
	$(PYTHON) -m ruff check .
	@echo Линтер завершен без ошибок

format-check:
	$(PYTHON) -m ruff format --check .
	@echo Форматирование проверено

format:
	$(PYTHON) -m ruff format .
	@echo Код отформатирован

typecheck:
	$(PYTHON) -m pyright
	@echo Проверка типов завершена

fullcheck: lint format-check typecheck
	@echo Все статические проверки пройдены

test:
	$(PYTHON) -m pytest -q

fulltest: fullcheck test
	@echo Все проверки и тесты пройдены

build:
	$(PYTHON) -m build

deploy:
	@echo deploy target - заглушка

precommit:
	$(PYTHON) -m pip install pre-commit
	$(PYTHON) -m pre_commit install
	$(PYTHON) -m pre_commit install --hook-type pre-push
	@echo Git hooks установлены

clean:
	rm -rf __pycache__ tests/__pycache__ .ruff_cache .pytest_cache .mypy_cache
	find . -name "*.pyc" -delete
	@echo Очистка завершена
