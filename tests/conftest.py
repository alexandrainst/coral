"""General functions and fixtures related to `pytest`."""

import sys
from typing import Generator

import pytest
from datasets import DatasetDict, IterableDatasetDict
from hydra import compose, initialize
from omegaconf import DictConfig

from coral_models.data import clean_dataset, load_data

# Initialise Hydra
initialize(config_path="../config", version_base=None)


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


@pytest.fixture(scope="session", params=["test_wav2vec2", "test_whisper"])
def cfg(request) -> Generator[DictConfig, None, None]:
    yield compose(
        config_name="config",
        overrides=[
            f"model={request.param}",
            "dataset=test_dataset",
            "fp16=false",
        ],
    )


@pytest.fixture(scope="session")
def dataset(cfg) -> Generator[DatasetDict | IterableDatasetDict, None, None]:
    yield load_data(cfg)


@pytest.fixture(scope="session")
def cleaned_dataset(
    cfg, dataset
) -> Generator[DatasetDict | IterableDatasetDict, None, None]:
    if cfg.model.clean_dataset:
        yield clean_dataset(cfg, dataset=dataset)
    else:
        yield dataset
