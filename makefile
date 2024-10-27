# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Create poetry env file if it does not already exist
ifeq (,$(wildcard ${HOME}/.poetry/env))
  $(shell mkdir ${HOME}/.poetry)
  $(shell touch ${HOME}/.poetry/env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Ensure that `pipx` and `poetry` will be able to run, since `pip` and `brew` put these
# in the following folders on Unix systems
export PATH := ${HOME}/.local/bin:/opt/homebrew/bin:$(PATH)

# Prevent DBusErrorResponse during `poetry install`
# (see https://stackoverflow.com/a/75098703 for more information)
export PYTHON_KEYRING_BACKEND := keyring.backends.null.Keyring

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'coral' project..."
	@$(MAKE) --quiet install-brew
	@$(MAKE) --quiet install-pipx
	@$(MAKE) --quiet install-poetry
	@$(MAKE) --quiet setup-poetry
	@$(MAKE) --quiet setup-environment-variables
	@echo "Installed the 'coral' project. If you want to use pre-commit hooks, run 'make install-pre-commit'."

install-brew:
	@if [ $$(uname) = "Darwin" ] && [ "$(shell which brew)" = "" ]; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Installed Homebrew."; \
	fi

install-pipx:
	@if [ "$(shell which pipx)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Linux*) installCmd='sudo apt install pipx'; ;; \
				(*Darwin*) installCmd='brew install pipx'; ;; \
				(*CYGWIN*) installCmd='py -3 -m pip install --upgrade --user pipx'; ;; \
				(*) installCmd='python3 -m pip install --upgrade --user pipx'; ;; \
			esac; \
			$${installCmd}; \
		pipx ensurepath --force; \
		echo "Installed pipx."; \
	fi

install-poetry:
	@if [ ! "$(shell poetry --version)" = "Poetry (version 1.8.2)" ]; then \
		python3 -m pip uninstall -y poetry poetry-core poetry-plugin-export; \
		pipx install --force poetry==1.8.2; \
		echo "Installed Poetry."; \
	fi

setup-poetry:
	@poetry env use python3.11 && poetry install --extras all

setup-environment-variables:
	@poetry run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@poetry run python src/scripts/fix_dot_env_file.py --non-interactive

docs:  ## Generate documentation
	@poetry run pdoc --docformat google src/coral -o docs
	@echo "Saved documentation."

view-docs:  ## View documentation
	@echo "Viewing API documentation..."
	@uname=$$(uname); \
		case $${uname} in \
			(*Linux*) openCmd='xdg-open'; ;; \
			(*Darwin*) openCmd='open'; ;; \
			(*CYGWIN*) openCmd='cygstart'; ;; \
			(*) echo 'Error: Unsupported platform: $${uname}'; exit 2; ;; \
		esac; \
		"$${openCmd}" docs/coral.html

test:  ## Run tests
	@poetry run pytest && poetry run readme-cov

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

install-pre-commit:  ## Install pre-commit hooks
	@poetry run pre-commit install

clean: lint format type-check  ## Lint, format, and type-check the code

lint:  ## Lint the code
	@poetry run ruff check . --fix

format:  ## Format the code
	@poetry run ruff format .

type-check:  ## Run type checking
	@poetry run mypy . \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--show-error-codes \
		--check-untyped-defs

check: lint format type-check  ## Check the code

roest-315m:  ## Train the Røst-315M model
	@OMP_NUM_THREADS=1 \
		accelerate launch \
		--use-deepspeed \
		src/scripts/finetune_asr_model.py \
		model=wav2vec2-small \
		datasets=[coral,common_voice_17] \
		dataset_probabilities=[0.95,0.05] \
		decoder_datasets=[wikipedia,common_voice,reddit] \
		push_to_hub=true \
		dataloader_num_workers=4 \
		model_id=roest-315m \
		private=true \
		per_device_batch_size=64

roest-809m:  ## Train the Røst-809M model
	@OMP_NUM_THREADS=1 \
		accelerate launch \
		--use-deepspeed \
		src/scripts/finetune_asr_model.py \
		model=whisper-large-turbo \
		datasets=[coral,common_voice_17] \
		dataset_probabilities=[0.95,0.05] \
		push_to_hub=true \
		dataloader_num_workers=4 \
		model_id=roest-809m \
		private=true \
		per_device_batch_size=64

roest-1b:  ## Train the Røst-1B model
	@OMP_NUM_THREADS=1 \
		accelerate launch \
		--use-deepspeed \
		src/scripts/finetune_asr_model.py \
		model=wav2vec2-medium \
		datasets=[coral,common_voice_17] \
		dataset_probabilities=[0.95,0.05] \
		decoder_datasets=[wikipedia,common_voice,reddit] \
		push_to_hub=true \
		dataloader_num_workers=4 \
		model_id=roest-1b \
		private=true \
		per_device_batch_size=64

roest-1.5b:  ## Train the Røst-1.5B model
	@OMP_NUM_THREADS=1 \
		accelerate launch \
		--use-deepspeed \
		src/scripts/finetune_asr_model.py \
		model=whisper-large \
		datasets=[coral,common_voice_17] \
		dataset_probabilities=[0.95,0.05] \
		push_to_hub=true \
		dataloader_num_workers=4 \
		model_id=roest-1.5b \
		private=true \
		per_device_batch_size=64

roest-2b:  ## Train the Røst-2B model
	@OMP_NUM_THREADS=1 \
		accelerate launch \
		--use-deepspeed \
		src/scripts/finetune_asr_model.py \
		model=wav2vec2-large \
		datasets=[coral,common_voice_17] \
		dataset_probabilities=[0.95,0.05] \
		decoder_datasets=[wikipedia,common_voice,reddit] \
		push_to_hub=true \
		dataloader_num_workers=4 \
		model_id=roest-2b \
		private=true \
		per_device_batch_size=64
