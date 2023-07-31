"""Unit tests for the `utils` module."""

import warnings

import datasets.utils.logging as ds_logging
import transformers.utils.logging as hf_logging
from datasets.utils import enable_progress_bar

from coral_models.utils import block_terminal_output, transformers_output_ignored


class output_blocked:
    """Convenience context manager to block terminal output."""

    def __enter__(self) -> None:
        block_terminal_output()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ds_logging.set_verbosity_warning()
        warnings.filterwarnings("default", category=UserWarning)
        enable_progress_bar()


class TestBlockTerminalOutput:
    def test_user_warnings_are_ignored(self) -> None:
        # Check that user warnings are raised by default
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("This is a user warning", category=UserWarning)
            assert len(w) == 1

        # Check that user warnings are ignored after `block_terminal_output` is called
        with output_blocked():
            with warnings.catch_warnings(record=True) as w:
                warnings.warn("This is a user warning", category=UserWarning)
                assert len(w) == 0

    def test_datasets_logging_level_is_error(self) -> None:
        assert ds_logging.get_verbosity() == ds_logging.WARNING
        with output_blocked():
            assert ds_logging.get_verbosity() == ds_logging.ERROR

    def test_datasets_progress_bars_are_disabled(self) -> None:
        assert ds_logging.is_progress_bar_enabled()
        with output_blocked():
            assert not ds_logging.is_progress_bar_enabled()


def test_transformers_output_ignored() -> None:
    assert hf_logging.get_verbosity() == hf_logging.INFO
    with transformers_output_ignored():
        assert hf_logging.get_verbosity() == hf_logging.ERROR
