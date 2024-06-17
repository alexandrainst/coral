"""General utility functions."""

import contextlib
import logging
import warnings
from functools import partialmethod
from pathlib import Path

import datasets.utils.logging as ds_logging
import tqdm as tqdm_package
import transformers.utils.logging as hf_logging
from datasets import Dataset, IterableDataset
from datasets.utils import disable_progress_bar
from tqdm.auto import tqdm

DIALECT_MAP: dict[str, str] = {
    "nordvestsjællandsk": "sjællandsk",
    "sydsjællandsk": "sjællandsk",
    "nordsjællandsk": "sjællandsk",
    "midtøstjysk": "østjysk",
    "vendsysselsk": "østjysk",
    "midtjysk": "østjysk",
    "nørrejysk": "østjysk",
    "vestlig sønderysk": "sønderjysk",
    "mellemslesvisk": "sønderjysk",
    "sydøstjysk": "sønderjysk",
    "amagermål": "københavnsk",
}


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

    # Disable the tokeniser progress bars from the `datasets` library
    disable_progress_bar()


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
    """Temporarily monkeypatch."""
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
    iterable_dataset: IterableDataset, dataset_id: str | None = None
) -> Dataset:
    """Convert an IterableDataset to a Dataset.

    Args:
        iterable_dataset:
            The IterableDataset to convert.
        dataset_id:
            The ID of the dataset, which is used to store and re-load the dataset.

    Returns:
        Dataset:
            The converted Dataset.
    """
    if dataset_id is not None:
        dataset_dir = Path.home() / ".cache" / "huggingface" / "datasets" / dataset_id
        if dataset_dir.exists():
            return Dataset.load_from_disk(str(dataset_dir))

    def gen_from_iterable_dataset():
        yield from tqdm(iterable=iterable_dataset)

    dataset = Dataset.from_generator(
        generator=gen_from_iterable_dataset, features=iterable_dataset.features
    )
    assert isinstance(dataset, Dataset)

    if dataset_id is not None:
        dataset_dir.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(str(dataset_dir))

    return dataset
