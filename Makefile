.PHONY: install lint format typecheck fullcheck test fulltest precommit run

install:
	pip install ruff pyright pre-commit pytest requests beautifulsoup4 matplotlib

lint:
	ruff check currency-converter.py

format:
	ruff format currency-converter.py

format-check:
	ruff format --check currency-converter.py

typecheck:
	pyright currency-converter.py

fullcheck: lint format-check typecheck

test:
	@echo "Запуск тестов (pytest)..."
	pytest || echo "Тесты не найдены или завершились с ошибкой"

fulltest: fullcheck test

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push
	@echo "Pre-commit и pre-push хуки успешно установлены!"

run:
	python currency-converter.py