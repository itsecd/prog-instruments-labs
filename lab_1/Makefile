.PHONY: install run test lint format format-check typecheck fullcheck fulltest precommit

install:
	python -m pip install -e ".[dev]"

run:
	python -c "from src.tennis import TennisGame; game = TennisGame('Player 1', 'Player 2'); print(game.score())"

test:
	python -m pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

format-check:
	ruff format --check src tests

typecheck:
	pyright

fullcheck: lint format-check typecheck

fulltest: fullcheck test

precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push
