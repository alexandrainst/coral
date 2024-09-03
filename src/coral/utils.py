"""General utility functions."""

import contextlib
import logging
import multiprocessing as mp
import warnings
from collections.abc import Callable
from functools import partialmethod
from pathlib import Path

import datasets.utils.logging as ds_logging
import tqdm as tqdm_package
import transformers.utils.logging as hf_logging
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    NamedSplit,
    disable_progress_bar,
    enable_progress_bar,
)
from tqdm.auto import tqdm

from coral.data import Data


def block_terminal_output() -> None:
    """Blocks undesired terminal output.

    This blocks the following output:
        - User warnings
        - Logs from the `datasets` package
        - Progress bars from the `datasets` package
    """
    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Disable logging from Hugging Face libraries
    ds_logging.set_verbosity_error()
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("pyctcdecode").setLevel(logging.ERROR)


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        hf_logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        hf_logging.set_verbosity_info()


@contextlib.contextmanager
def monkeypatched(obj, name, patch):
    """Temporarily monkeypatch.

    Args:
        obj:
            The object to monkeypatch.
        name:
            The name of the attribute to monkeypatch.
        patch:
            The patch to apply.
    """
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


@contextlib.contextmanager
def disable_tqdm():
    """Context manager to disable tqdm."""

    def _patch(old_init):
        return partialmethod(old_init, disable=True)

    with monkeypatched(tqdm_package.std.tqdm, "__init__", _patch):
        yield


def convert_iterable_dataset_to_dataset(
    iterable_dataset: IterableDataset,
    split_name: str = "train",
    dataset_id: str | None = None,
) -> Dataset:
    """Convert an IterableDataset to a Dataset.

    Args:
        iterable_dataset:
            The IterableDataset to convert.
        split_name (optional):
            The name of the split. Defaults to "train".
        dataset_id (optional):
            The ID of the dataset, which is used to store and re-load the dataset. If
            None then the dataset is not stored. Defaults to None.

    Returns:
        The converted Dataset.
    """
    if dataset_id is not None:
        dataset_dir = Path.home() / ".cache" / "huggingface" / "datasets" / dataset_id
        if dataset_dir.exists():
            return Dataset.load_from_disk(str(dataset_dir))

    splits_info = iterable_dataset.info.splits
    num_examples = None if splits_info is None else splits_info[split_name].num_examples

    def gen_from_iterable_dataset():
        yield from tqdm(
            iterable=iterable_dataset,
            total=num_examples,
            desc="Converting iterable dataset to regular dataset",
        )

    with no_datasets_progress_bars():
        dataset = Dataset.from_generator(
            generator=gen_from_iterable_dataset,
            features=iterable_dataset.features,
            split=NamedSplit(name=split_name),
            num_proc=mp.cpu_count(),
        )
    assert isinstance(dataset, Dataset)

    if dataset_id is not None:
        dataset_dir.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(str(dataset_dir))

    return dataset


class no_datasets_progress_bars:
    """Context manager that disables the `datasets` progress bars."""

    def __enter__(self):
        """Disable the progress bar."""
        disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Re-enable the progress bar."""
        enable_progress_bar()


def recover_dataset_info(new_dataset: Data, old_dataset: Data) -> Data:
    """Recover the dataset info from the old dataset.

    This is used after calling `map` on a dataset, as that operation removes the
    dataset info.

    Args:
        new_dataset:
            The new dataset.
        old_dataset:
            The old dataset.

    Returns:
        The new dataset with the recovered info.
    """
    if isinstance(old_dataset, Dataset | IterableDataset) and isinstance(
        new_dataset, Dataset | IterableDataset
    ):
        new_dataset._info = old_dataset._info

    elif isinstance(old_dataset, DatasetDict | IterableDatasetDict) and isinstance(
        new_dataset, DatasetDict | IterableDatasetDict
    ):
        for key, value in old_dataset.items():
            if key in new_dataset:
                new_dataset[key]._info = value._info

    return new_dataset


def map_with_info(dataset: Data, function: Callable, **map_kwargs) -> Data:
    """Map a function over a dataset, preserving the dataset info.

    Args:
        dataset:
            The dataset to map over.
        function:
            The function to map.
        map_kwargs:
            Additional keyword arguments to pass to the `map` method.

    Returns:
        The mapped dataset.
    """
    if not isinstance(dataset, Dataset | DatasetDict):
        map_kwargs.pop("num_proc")
        map_kwargs.pop("batched", None)
        map_kwargs.pop("batch_size", None)
        map_kwargs.pop("desc", None)
    mapped = dataset.map(function=function, **map_kwargs)
    mapped = recover_dataset_info(new_dataset=mapped, old_dataset=dataset)
    return mapped


def filter_with_info(dataset: Data, function: Callable, **filter_kwargs) -> Data:
    """Filter a dataset, preserving the dataset info.

    Args:
        dataset:
            The dataset to filter.
        function:
            The function to filter.
        filter_kwargs:
            Additional keyword arguments to pass to the `filter` method.

    Returns:
        The filtered dataset.
    """
    if not isinstance(dataset, Dataset | DatasetDict):
        filter_kwargs.pop("num_proc")
        filter_kwargs.pop("batched", None)
        filter_kwargs.pop("batch_size", None)
        filter_kwargs.pop("desc", None)
    filtered = dataset.filter(function=function, **filter_kwargs)
    filtered = recover_dataset_info(new_dataset=filtered, old_dataset=dataset)
    return filtered
