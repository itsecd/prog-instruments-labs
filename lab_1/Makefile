PY := py

.PHONY: help lint format format-check typecheck fullcheck test fulltest run run-gui precommit clean

help:
	@echo "Доступные команды:"
	@echo "  make lint          - ruff check"
	@echo "  make format        - ruff format"
	@echo "  make format-check  - ruff format --check"
	@echo "  make typecheck     - pyright"
	@echo "  make fullcheck     - lint + format-check + typecheck"
	@echo "  make test          - unit-тесты"
	@echo "  make fulltest      - fullcheck + test"
	@echo "  make run           - запуск CLI (--help)"
	@echo "  make run-gui       - запуск GUI"
	@echo "  make precommit     - установка git hooks"
	@echo "  make clean         - очистка кэшей"

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

format-check:
	$(PY) -m ruff format . --check

typecheck:
	$(PY) -m pyright

fullcheck: lint format-check typecheck

test:
	$(PY) -m unittest test_hash -v

fulltest: fullcheck test

run:
	$(PY) cli.py --help

run-gui:
	$(PY) gui.py

precommit:
	$(PY) -m pre_commit install
	$(PY) -m pre_commit install --hook-type pre-push

clean:
	-rmdir /s /q __pycache__ .ruff_cache .pytest_cache 2>nul