"""General functions and fixtures related to `pytest`."""

import itertools as it
import sys
from typing import Generator

import pytest
from coral.data import load_data
from datasets import DatasetDict, IterableDatasetDict
from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import DictConfig

load_dotenv()


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
            ["test-wav2vec2", "test-whisper"],
            ["test_dataset", "[test_dataset,test_dataset]"],
        )
    ),
    ids=lambda x: f"model: {x[0]}, dataset: {x[1]}",
)
def config(request) -> Generator[DictConfig, None, None]:
    """Hydra configuration."""
    model, datasets = request.param
    yield compose(
        config_name="config",
        overrides=[
            f"model={model}",
            f"datasets={datasets}",
            "fp16=false",
            "total_batch_size=2",
            "per_device_batch_size=2",
            "max_steps=2",
            "save_total_limit=0",
        ],
    )


@pytest.fixture(scope="session")
def dataset(config) -> Generator[DatasetDict | IterableDatasetDict, None, None]:
    """ASR Dataset."""
    yield load_data(config=config)
