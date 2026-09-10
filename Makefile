run:
	python lab_1/parrot.py

tests:
	pytest lab_1\test_parrot.py

format:
	ruff format . --check

check:
	ruff check .

types:
	pyright .

fix:
	ruff check . --fix

format_fix:
	ruff format .

fullcheck: tests check format types

fullfix: fix format_fix

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push
