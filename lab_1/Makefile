.PHONY: help venv install run lint format format-check typecheck fullcheck test fulltest build deploy precommit clean

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
	python -m venv venv

install:
	python -m pip install --upgrade pip
	python -m pip install ruff pyright pre-commit pytest build

run:
	python script.py text.txt alfa.txt key.txt decrypted.txt

lint:
	python -m ruff check .

format-check:
	python -m ruff format --check .

format:
	python -m ruff format .

typecheck:
	python -m mypy script.py

fullcheck: lint format-check typecheck

test:
	python -m pytest -q

fulltest: fullcheck test

build:
	python -m build

deploy:
	@echo deploy target - заглушка

precommit:
	python -m pip install pre-commit
	python -m pre_commit install
	python -m pre_commit install --hook-type pre-push

clean:
	rm -rf __pycache__ tests/__pycache__ .ruff_cache .pytest_cache .mypy_cache
	find . -name "*.pyc" -delete
