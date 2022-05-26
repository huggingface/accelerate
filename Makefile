.PHONY: quality style test docs

check_dirs := tests src examples

# Check that source code meets quality standards

extra_quality_checks:
	python utils/check_copies.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/style_doc.py src/accelerate docs/source --max_len 119

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	python utils/style_doc.py src/accelerate docs/source --max_len 119 --check_only

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	black $(check_dirs)
	isort $(check_dirs)
	python utils/style_doc.py src/accelerate docs/source --max_len 119
	
# Run tests for the library
test:
	python -m pytest -s -v ./tests/ --ignore=./tests/test_examples.py --ignore=./tests/test_scheduler.py --ignore=./tests/test_cpu.py
	python -m pytest -s -sv ./tests/test_cpu.py ./tests/test_scheduler.py

test_examples:
	python -m pytest -s -v ./tests/test_examples.py
