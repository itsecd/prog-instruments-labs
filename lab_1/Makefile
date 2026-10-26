.PHONY: install run lint format format-check typecheck fullcheck precommit

install:
	python -m pip install -r requirements.txt

run:
	python laba1.py data.txt

lint:
	python -m ruff check laba1.py

format:
	python -m ruff format laba1.py

format-check:
	python -m ruff format --check laba1.py

typecheck:
	python -m pyright

fullcheck: lint format-check typecheck

precommit:
	python -X utf8 -m pre_commit install
	python -X utf8 -m pre_commit install --hook-type pre-push
