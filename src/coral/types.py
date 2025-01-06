"""Types used in the project."""

from typing import TypeVar

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

Data = TypeVar(
    "Data", bound=Dataset | IterableDataset | DatasetDict | IterableDatasetDict
)
