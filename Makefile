.PHONY: quality style test docs

check_dirs := tests src

# Check that source code meets quality standards

extra_quality_checks:
	python utils/check_copies.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/style_doc.py src/transformers docs/source --max_len 119

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	black $(check_dirs)
	isort $(check_dirs)
	
# Run tests for the library
test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Check that docs can build
docs:
	cd docs && make html SPHINXOPTS="-W"
