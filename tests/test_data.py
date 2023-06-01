"""Unit tests for the `data` module."""

from datasets import DatasetDict, IterableDatasetDict


class TestLoadData:
    def test_dataset_type(self, dataset) -> None:
        assert isinstance(dataset, DatasetDict) or isinstance(
            dataset, IterableDatasetDict
        )

    def test_splits_are_in_dataset(self, dataset) -> None:
        assert "train" in dataset
        assert "val" in dataset
        assert "test" in dataset
