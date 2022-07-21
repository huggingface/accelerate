.PHONY: quality style test docs

check_dirs := tests src examples benchmarks

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
	python -m pytest -s -v ./tests/ --ignore=./tests/test_examples.py --durations=0

test_big_modeling:
	python -m pytest -s -v ./tests/test_big_modeling.py --durations=0

test_core:
	python -m pytest -s -v ./tests/ --ignore=./tests/test_examples.py --ignore=./tests/deepspeed --ignore=./tests/test_big_modeling.py --durations=0

test_deepspeed:
	python -m pytest -s -v ./tests/deepspeed --durations=0

test_examples:
	python -m pytest -s -v ./tests/test_examples.py --durations=0

# Broken down example tests for the CI runners
test_example_differences:
	python -m pytest -s -v ./tests/test_examples.py::ExampleDifferenceTests --durations=0

test_checkpoint_epoch:
	python -m pytest -s -v ./tests/test_examples.py::FeatureExamplesTests -k "by_epoch" --durations=0

test_checkpoint_step:
	python -m pytest -s -v ./tests/test_examples.py::FeatureExamplesTests -k "by_step" --durations=0

# Same as test but used to install only the base dependencies
test_prod:
	$(MAKE) test_core

test_rest:
	python -m pytest -s -v ./tests/test_examples.py::FeatureExamplesTests -k "not by_step and not by_epoch" --durations=0