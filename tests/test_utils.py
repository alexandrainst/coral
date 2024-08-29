"""Unit tests for the `utils` module."""

import datasets.utils.logging as ds_logging
import transformers.utils.logging as hf_logging

from coral.utils import block_terminal_output, transformers_output_ignored


class output_blocked:
    """Convenience context manager to block terminal output."""

    def __enter__(self) -> None:
        """Block terminal output."""
        block_terminal_output()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unblock terminal output."""
        ds_logging.set_verbosity_warning()


class TestBlockTerminalOutput:
    """Tests for the `block_terminal_output` function."""

    def test_datasets_logging_level_is_error(self) -> None:
        """Test that the datasets logging level is set to error."""
        ds_logging.set_verbosity_warning()
        assert ds_logging.get_verbosity() == ds_logging.WARNING
        with output_blocked():
            assert ds_logging.get_verbosity() == ds_logging.ERROR


def test_transformers_output_ignored() -> None:
    """Test that the transformers output is ignored."""
    hf_logging.set_verbosity_info()
    assert hf_logging.get_verbosity() == hf_logging.INFO
    with transformers_output_ignored():
        assert hf_logging.get_verbosity() == hf_logging.ERROR
