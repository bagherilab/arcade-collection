.PHONY: clean build docs

clean: # clean all build, python, and testing files
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -rf .tox/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache

build: # run tox tests and lint
	tox

docs: # generates documentation
	make -C docs html

release:
	poetry version $(version)
	git checkout -b release/v$$(poetry version -s)
	git add pyproject.toml
	git commit -m "Bump version → $$(poetry version -s)"
