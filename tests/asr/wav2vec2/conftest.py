"""ASR-specific functions and fixtures related to `pytest`."""

from typing import Generator

import pytest
from datasets import DatasetDict, IterableDatasetDict

from coral_models.asr.wav2vec2.clean import clean_dataset


@pytest.fixture(scope="module")
def cleaned_dataset(
    cfg, dataset
) -> Generator[DatasetDict | IterableDatasetDict, None, None]:
    yield clean_dataset(cfg, dataset=dataset)
