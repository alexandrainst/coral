"""Unit tests for the `utils` module."""

import warnings

import datasets.utils.logging as ds_logging
import transformers.utils.logging as hf_logging
from datasets.utils import enable_progress_bar

from coral_models.utils import block_terminal_output, ignore_transformers_output


class blocking_output:
    """Convenience context manager to block terminal output."""

    def __enter__(self) -> None:
        block_terminal_output()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ds_logging.set_verbosity_warning()
        warnings.filterwarnings("default", category=UserWarning)
        enable_progress_bar()


class TestBlockTerminalOutput:
    """Tests for the `block_terminal_output` function."""

    def test_user_warnings_are_ignored(self) -> None:
        """Test that user warnings are ignored."""

        # Check that user warnings are raised by default
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("This is a user warning", category=UserWarning)
            assert len(w) == 1

        # Check that user warnings are ignored after `block_terminal_output` is called
        with blocking_output():
            with warnings.catch_warnings(record=True) as w:
                warnings.warn("This is a user warning", category=UserWarning)
                assert len(w) == 0

    def test_datasets_logging_level_is_error(self) -> None:
        """Test that the `datasets` logging level is set to `ERROR`."""

        # Check that the `datasets` logging level is set to `WARNING` by default
        assert ds_logging.get_verbosity() == ds_logging.WARNING

        # Check that the `datasets` logging level is set to `ERROR` after
        # `block_terminal_output` is called
        with blocking_output():
            assert ds_logging.get_verbosity() == ds_logging.ERROR

    def test_datasets_progress_bars_are_disabled(self) -> None:
        """Test that the `datasets` progress bars are disabled."""

        # Check that the `datasets` progress bars are enabled by default
        assert ds_logging.is_progress_bar_enabled()

        # Check that the `datasets` progress bars are disabled after
        # `block_terminal_output` is called
        with blocking_output():
            assert not ds_logging.is_progress_bar_enabled()


def test_ignore_transformers_output() -> None:
    """Test that the `ignore_transformers_output` context manager works."""

    # Check that the `transformers` logging level is set to `WARNING` by default
    assert hf_logging.get_verbosity() == hf_logging.WARNING

    # Check that the `transformers` logging level is set to `ERROR` after
    # `ignore_transformers_output` is called
    with ignore_transformers_output():
        assert hf_logging.get_verbosity() == hf_logging.ERROR
