"""General utility functions."""

import contextlib
import logging
import warnings
from functools import partialmethod

import datasets.utils.logging as ds_logging
import tqdm
import transformers.utils.logging as hf_logging
from datasets.utils import disable_progress_bar


def block_terminal_output() -> None:
    """Blocks undesired terminal output.

    This blocks the following output:
        - User warnings
        - Logs from the `datasets` package
        - Progress bars from the `datasets` package
    """

    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)

    # Disable logging from Hugging Face libraries
    ds_logging.set_verbosity_error()
    logging.getLogger("accelerate").setLevel(logging.ERROR)

    # Disable the tokeniser progress bars from the `datasets` library
    disable_progress_bar()


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        hf_logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

    with monkeypatched(tqdm.std.tqdm, "__init__", _patch):
        yield
