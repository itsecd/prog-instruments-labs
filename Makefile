setup: requirements run

requirements:
	pip	install -r requirements.txt

run:
	python main.py

fullcheck: lint format typecheck

lint:
	-ruff check

format:
	-ruff format --check

typecheck:
	-pyright

