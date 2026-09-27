FIXED_FILE = fixed_tennis.py

.PHONY: lint lintfix formatcheck format typecheck fullcheck run precommit

lint:
	@ruff check $(FIXED_FILE)

lintfix:
	@ruff check --fix $(FIXED_FILE)

formatcheck:
	@ruff format --check $(FIXED_FILE)

format:
	@ruff format $(FIXED_FILE)

typecheck:
	@pyright $(FIXED_FILE)

fullcheck: lintfix format typecheck

run:
	@python $(FIXED_FILE)

precommit:
	@pre-commit install --hook-type pre-commit
	@pre-commit install --hook-type pre-push