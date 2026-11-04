.PHONY: lint format format-check typecheck fullcheck test fulltest run precommit clean

run:
	python lab_1/main.py

lint:
	ruff check lab_1/

format:
	ruff format lab_1/

format-check:
	ruff format --check lab_1/

typecheck:
	pyright lab_1/

fullcheck: lint format-check typecheck

test:
	pytest tests/ -v

fulltest: fullcheck test

precommit:
	pre-commit install
	pre-commit install --hook-type pre-push

precommit-run:
	pre-commit run --all-files

clean:
	Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
	Get-ChildItem -Recurse -Filter *.pyc | Remove-Item -Force
