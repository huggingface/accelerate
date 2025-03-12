.PHONY: quality style test docs utils

check_dirs := .

# Check that source code meets quality standards

extra_quality_checks:
	python utils/check_copies.py
	python utils/check_dummies.py
	python utils/check_repo.py
	doc-builder style src/accelerate docs/source --max_len 119

# this target runs checks on all files
quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)
	doc-builder style src/accelerate docs/source --max_len 119 --check_only

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
	doc-builder style src/accelerate docs/source --max_len 119
	
# Run tests for the library
test_big_modeling:
	python -m pytest -s -v ./tests/test_big_modeling.py ./tests/test_modeling_utils.py $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_big_modeling.log",)

test_core:
	python -m pytest -s -v ./tests/ --ignore=./tests/test_examples.py --ignore=./tests/deepspeed --ignore=./tests/test_big_modeling.py \
	--ignore=./tests/fsdp --ignore=./tests/tp --ignore=./tests/test_cli.py $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_core.log",)

test_cli:
	python -m pytest -s -v ./tests/test_cli.py $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_cli.log",)

test_deepspeed:
	python -m pytest -s -v ./tests/deepspeed $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_deepspeed.log",)

test_fsdp:
	python -m pytest -s -v ./tests/fsdp $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_fsdp.log",)

test_tp:
	python -m pytest -s -v ./tests/tp $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_tp.log",)

# Since the new version of pytest will *change* how things are collected, we need `deepspeed` to 
# run after test_core and test_cli
test:
	$(MAKE) test_core
	$(MAKE) test_cli
	$(MAKE) test_big_modeling
	$(MAKE) test_deepspeed
	$(MAKE) test_fsdp
	$(MAKE) test_tp

test_examples:
	python -m pytest -s -v ./tests/test_examples.py $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_examples.log",)

# Broken down example tests for the CI runners
test_integrations:
	python -m pytest -s -v ./tests/deepspeed ./tests/fsdp ./tests/tp $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_integrations.log",)

test_example_differences:
	python -m pytest -s -v ./tests/test_examples.py::ExampleDifferenceTests $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_example_diff.log",)

test_checkpoint_epoch:
	python -m pytest -s -v ./tests/test_examples.py::FeatureExamplesTests -k "by_epoch" $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_checkpoint_epoch.log",)

test_checkpoint_step:
	python -m pytest -s -v ./tests/test_examples.py::FeatureExamplesTests -k "by_step" $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_checkpoint_step.log",)

# Same as test but used to install only the base dependencies
test_prod:
	$(MAKE) test_core

test_rest:
	python -m pytest -s -v ./tests/test_examples.py::FeatureExamplesTests -k "not by_step and not by_epoch" $(if $(IS_GITHUB_CI),--report-log "$(PYTORCH_VERSION)_rest.log",)

# For developers to prepare a release
prepare_release:
	rm -rf dist build
	python setup.py bdist_wheel sdist

# Make sure this is ran in a fresh venv of some form
install_test_release:
	pip uninstall accelerate -y
	pip install -i https://testpypi.python.org/pypi --extra-index-url https://pypi.org/simple accelerate$(if $(version),==$(version),)

# Run as `make target=testpypi upload_release`
upload_release:
	@if [ "$(target)" != "testpypi" ] && [ "$(target)" != "pypi" ]; then \
		echo "Error: target must be either 'testpypi' or 'pypi'"; \
		exit 1; \
	fi
	twine upload dist/* -r $(target)