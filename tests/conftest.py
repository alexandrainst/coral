"""General functions and fixtures related to `pytest`."""

import itertools as it
import os
import sys
from collections.abc import Generator

import pytest
from datasets import Dataset, load_dataset
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
def finetuning_config(request) -> Generator[DictConfig, None, None]:
    """Hydra configuration."""
    model, datasets = request.param
    yield compose(
        config_name="asr_finetuning",
        overrides=[
            f"model={model}",
            f"datasets={datasets}",
            "bf16_allowed=false",
            "fp16_allowed=false",
            "total_batch_size=2",
            "per_device_batch_size=2",
            "max_steps=2",
            "save_total_limit=0",
        ],
    )


@pytest.fixture(scope="session")
def dataset(finetuning_config) -> Generator[Dataset, None, None]:
    """Load the dataset for testing."""
    dataset_config = list(finetuning_config.datasets.values())[0]
    dataset = load_dataset(
        path=dataset_config.id,
        name=dataset_config.subset,
        split=dataset_config.train_name,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        trust_remote_code=True,
    )
    assert isinstance(dataset, Dataset)
    if dataset_config.text_column != "text":
        dataset = dataset.rename_column(dataset_config.text_column, "text")
    if dataset_config.audio_column != "audio":
        dataset = dataset.rename_column(dataset_config.audio_column, "audio")
    yield dataset
