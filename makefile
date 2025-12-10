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

# Force the installation to use position-independent code, which helps with the
# installation of the `samplerate` package, as it relies on an underlying C library.
export CFLAGS := -fPIC $(CFLAGS)
export CXXFLAGS := -fPIC $(CXXFLAGS)

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
	@if [ "$(shell uname)" = "Darwin" ]; then \
		uv sync --python 3.11; \
	else \
		uv sync --python 3.11 --all-extras; \
	fi

setup-environment-variables:
	@uv run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@uv run python src/scripts/fix_dot_env_file.py --non-interactive

test:  ## Run tests
	@uv run pytest && uv run readme-cov

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

check:  ## Lint, format, and type-check the code
	@uv run pre-commit run --all-files

roest-315m:  ## Train the Røst-315M model
	@OMP_NUM_THREADS=1 \
		uv run accelerate launch \
		--use-deepspeed \
		--zero-stage 2 \
		src/scripts/finetune_asr_model.py \
		model=wav2vec2-small \
		push_to_hub=true \
		model_id=roest-wav2vec2-315m-100k-steps \
		private=true \
		per_device_batch_size=64 \
		max_steps=100000

roest-1.5b:  ## Train the Røst-1.5B model
	@OMP_NUM_THREADS=1 \
		uv run accelerate launch \
		--use-deepspeed \
		--zero-stage 2 \
		src/scripts/finetune_asr_model.py \
		model=whisper-large \
		push_to_hub=true \
		model_id=roest-whisper-1.5b-30k-steps \
		private=true \
		per_device_batch_size=64 \
		max_steps=30000
