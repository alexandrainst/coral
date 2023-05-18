"""Unit tests for the `data` module."""

from datasets import DatasetDict, IterableDatasetDict


class TestLoadData:
    """Unit tests for the `load_data` function."""

    def test_dataset_type(self, dataset) -> None:
        """Test that the dataset is of the correct type."""
        assert isinstance(dataset, DatasetDict) or isinstance(
            dataset, IterableDatasetDict
        )

    def test_splits_are_in_dataset(self, dataset) -> None:
        """Test that the splits are in the dataset."""
        assert "train" in dataset
        assert "val" in dataset
        assert "test" in dataset
