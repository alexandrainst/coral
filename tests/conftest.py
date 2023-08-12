"""General functions and fixtures related to `pytest`."""

import itertools as it
import sys
from typing import Generator

import pytest
from datasets import DatasetDict, IterableDatasetDict
from hydra import compose, initialize
from omegaconf import DictConfig

from coral_models.data import load_data

# Initialise Hydra
initialize(config_path="../config", version_base=None)


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


@pytest.fixture(
    scope="session",
    params=list(
        it.product(
            ["test_wav2vec2", "test_whisper"],
            ["test_dataset", "[test_dataset,test_dataset]"],
        )
    ),
    ids=lambda x: f"model: {x[0]}, dataset: {x[1]}",
)
def cfg(request) -> Generator[DictConfig, None, None]:
    model, datasets = request.param
    yield compose(
        config_name="config",
        overrides=[
            f"model={model}",
            f"datasets={datasets}",
            "fp16=false",
        ],
    )


@pytest.fixture(scope="session")
def dataset(cfg) -> Generator[DatasetDict | IterableDatasetDict, None, None]:
    yield load_data(cfg)
