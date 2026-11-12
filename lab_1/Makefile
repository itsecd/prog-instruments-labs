.PHONY: help run lint lint-fix format format-check typecheck fullcheck clean

PYTHON ?= python

help:  ## Показать список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*##/ -/'

run:  ## Запустить симуляцию (создаёт картинки и видео в output/)
	mkdir -p output
	$(PYTHON) GameOfLife.py

lint:  ## Статический анализ (ruff check)
	ruff check .

lint-fix:  ## Статический анализ с автоисправлением
	ruff check --fix .

format:  ## Автоматически отформатировать код (ruff format)
	ruff format .

format-check:  ## Проверить форматирование без изменений
	ruff format --check .

typecheck:  ## Проверка типов (pyright)
	pyright

fullcheck: lint format-check typecheck  ## Все статические проверки

clean:  ## Удалить кэши и результаты запуска
	rm -rf output .ruff_cache __pycache__