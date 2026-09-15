run:
	python3 main.py -f data.txt -o output.txt

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

precommit:
	pre-commit install --hook-type pre-commit --hook-type pre-push

prepush:
	pre-commit run --all-files