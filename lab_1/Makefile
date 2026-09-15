run:
	python -m main.py -a 3

test:
	python -m unittest discover -s tests/ -p "*_test.py"

build: # Представим, что тут сборка через какой-нибудь pyinstaller

deploy: # Куда?

lint:
	ruff check .

format:
	ruff format .

typecheck:
	pyright .

fullcheck:
	ruff check .
	pyright .
	ruff format --check

fulltest:
	ruff check .
	pyright .
	ruff format --check
	python -m unittest discover -s tests/ -p "*_test.py"

precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push
