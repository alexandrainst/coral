# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'CoRal' project..."
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git
	@$(MAKE) --quiet install-pre-commit
	@echo "Installed the 'CoRal' project! You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a 'uv' project. Use 'uv add <package>' to install new dependencies and 'uv remove <package>' to remove them."

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		if [ "$(shell which rustup)" = "" ]; then \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			echo "Installed Rust."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update || true; \
	fi

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

install-dependencies:
	@uv python install 3.11
	@uv sync --python 3.11

setup-environment-variables:
	@uv run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@uv run python src/scripts/fix_dot_env_file.py --non-interactive

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name "${GIT_NAME}"
	@git config --local user.email "${GIT_EMAIL}"

test:  ## Run tests
	@uv run pytest && uv run readme-cov

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

lint:  ## Lint the project
	uv run ruff check . --fix --unsafe-fixes

format:  ## Format the project
	uv run ruff format .

type-check:  ## Type-check the project
	@uv run mypy . \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--show-error-codes \
		--check-untyped-defs

check: lint format type-check  ## Lint, format, and type-check the code

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
